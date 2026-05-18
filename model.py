import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True, dropout=0.0):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_drop = self.dropout(seq)
        seq_fts = self.fc(seq_drop)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
            
        # Add residual/skip connection to preserve ego features (crucial for heterophilic datasets like Amazon)
        out = out + seq_fts

        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Discriminator(nn.Module):
    """Attention-based Hybrid Discriminator.
    
    Learns to distinguish anomalies by dynamically fusing Semantic (Embedding) 
    and Structural (Residual) features.
    """

    def __init__(self, n_h, negsamp_round=None):
        super(Discriminator, self).__init__()
        self.n_h = n_h
        
        # Feature extraction for Embedding and Residual
        self.feat_emb = nn.Sequential(nn.Linear(n_h, n_h), nn.LeakyReLU(0.2))
        self.feat_resid = nn.Sequential(nn.Linear(n_h, n_h), nn.LeakyReLU(0.2))
        
        # Attention mechanism: [Emb, Resid] -> Weights
        self.att = nn.Linear(n_h * 2, 2)
        
        # Classifier head
        self.fc_out = nn.Linear(n_h, 1)
        
    def forward(self, inputs):
        # inputs: (1, N, 2*D) -> (N, 2*D)
        x = inputs.squeeze(0)
        
        # Split into Embedding and Residual
        emb = x[:, :self.n_h]
        resid = x[:, self.n_h:]
        
        # Transform features
        h_emb = self.feat_emb(emb)
        h_resid = self.feat_resid(resid)
        
        # Calculate Attention Weights
        concat = torch.cat([h_emb, h_resid], dim=1)
        weights = F.softmax(self.att(concat), dim=1)
        w_emb = weights[:, 0].unsqueeze(1)
        w_resid = weights[:, 1].unsqueeze(1)
        
        # Weighted Fusion
        fused = w_emb * h_emb + w_resid * h_resid
        
        # Predict Logits
        logits = self.fc_out(fused)
        return logits.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, dropout=0.0):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, n_h, activation, dropout=dropout)
        self.gcn2 = GCN(n_h, n_h, activation, dropout=dropout)
        self.gcn3 = GCN(n_h, n_h, activation, dropout=dropout)
        self.fc1 = nn.Linear(n_h, int(n_h / 2), bias=False)
        self.fc2 = nn.Linear(int(n_h / 2), int(n_h / 4), bias=False)
        self.fc3 = nn.Linear(int(n_h / 4), 1, bias=False)
        
        # Generator MLP
        self.gen_mlp = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_h, n_h)
        )
        
        # Predictor MLP (for Learned Residuals)
        # Decoupled from Generator to avoid objective conflict
        self.predictor = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_h, n_h)
        )

        # Residual prototypes (AnomalyGFM prompts) trained jointly
        self.normal_proto = nn.Parameter(torch.zeros(n_h))
        self.abnormal_proto = nn.Parameter(torch.zeros(n_h))
        nn.init.normal_(self.normal_proto, std=1e-2)
        nn.init.normal_(self.abnormal_proto, std=1e-2)

        # Initialize last layer to 0 to start with small perturbation if we use residual connection
        # But here we use direct mapping.
        
        self.fc6 = nn.Linear(n_h, n_h, bias=False)
        self.fc5 = nn.Linear(n_h, n_in, bias=False)
        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj_prop, adj_resid, generator_seed_idx, normal_idx, abnormal_idx, train_flag, args, sparse=False):
        """Run GCN forward pass and compute residual-aware logits.

        adj_prop: normalized adjacency (with self-loops) for message passing.
        adj_resid: row-normalized adjacency without self-loops for residuals.
        """
        h_1 = self.gcn1(seq1, adj_prop, sparse)
        emb = self.gcn2(h_1, adj_prop, sparse)
        
        # Normalize embeddings to prevent explosion and stabilize residual calculation
        emb = F.normalize(emb, p=2, dim=-1)

        emb_con = None
        emb_combine = None
        resid_fake = None
        proto_normal = self.normal_proto.unsqueeze(0)
        proto_abnormal = self.abnormal_proto.unsqueeze(0)

        # Gather normal / abnormal embeddings for downstream losses
        emb_normals = emb[:, normal_idx, :]
        emb_abnormal_real = emb[:, abnormal_idx, :] if abnormal_idx is not None and len(abnormal_idx) > 0 else None
        emb_abnormal = emb_abnormal_real

        if train_flag:
            # Use normals as seeds for generator to synthesize hard anomalies
            neigh_adj_seed = adj_resid[0, generator_seed_idx, :]

            # Generator input: neighborhood aggregation of seed normals
            seed_context = torch.mm(neigh_adj_seed, emb[0, :, :])  # (n_seed, D)

            # Inject gaussian noise to encourage diversity
            gen_noise = torch.randn_like(seed_context) * args.var + args.mean
            seed_context = seed_context + gen_noise

            emb_con = self.gen_mlp(seed_context)

            # build combined batch: normals followed by generated fakes
            # Residuals for normals (prototype space)
            neigh_agg_normals = torch.mm(adj_resid[0, normal_idx, :], emb[0, :, :])
            pred_normals = self.predictor(neigh_agg_normals)
            resid_normals = emb_normals.squeeze(0) - pred_normals

            # Residuals for generator outputs (using seed contexts)
            pred_fakes = self.predictor(seed_context)
            resid_fakes = emb_con - pred_fakes

            # Residuals for real anomalies if available
            if emb_abnormal_real is not None:
                neigh_agg_abn = torch.mm(adj_resid[0, abnormal_idx, :], emb[0, :, :])
                pred_abn = self.predictor(neigh_agg_abn)
                resid_abn = emb_abnormal_real.squeeze(0) - pred_abn
                emb_blocks = [emb_normals, torch.unsqueeze(emb_con, 0), emb_abnormal_real]
                resid_blocks = [resid_normals, resid_fakes, resid_abn]
            else:
                resid_abn = None
                emb_blocks = [emb_normals, torch.unsqueeze(emb_con, 0)]
                resid_blocks = [resid_normals, resid_fakes]

            emb_combine = torch.cat(emb_blocks, dim=1)
            resid_comb = torch.cat(resid_blocks, dim=0).unsqueeze(0)
            disc_input = torch.cat([emb_combine, resid_comb], dim=-1)
            f_3 = self.disc(disc_input)

            resid_fake = resid_fakes.unsqueeze(0)
        else:
            # inference: compute residual for all nodes and score by prototype
            neigh_agg_all = torch.mm(adj_resid[0, :, :], emb[0, :, :])  # (N, D)
            
            # Use Predictor to predict centers
            pred_all = self.predictor(neigh_agg_all)
            
            resid_all = (emb.squeeze(0) - pred_all).unsqueeze(0)  # (1,N,D)
            
            # Discriminator input: Hybrid Input [Embedding, Residual]
            disc_input = torch.cat([emb, resid_all], dim=-1)
            f_3 = self.disc(disc_input)

        # Also compute residuals for all / abnormal nodes consistently with predictor
        neigh_agg_all = torch.mm(adj_resid[0, :, :], emb[0, :, :])
        pred_all_full = self.predictor(neigh_agg_all)
        resid_all = (emb.squeeze(0) - pred_all_full).unsqueeze(0)

        if abnormal_idx is not None and len(abnormal_idx) > 0:
            neigh_agg_abn = neigh_agg_all[abnormal_idx, :]
            pred_abn = self.predictor(neigh_agg_abn)
            resid_abnormal = (emb_abnormal_real.squeeze(0) - pred_abn).unsqueeze(0)
        else:
            resid_abnormal = None

        return (emb,
            emb_combine,
            f_3,
            emb_con,
            emb_abnormal,
            resid_all,
            resid_fake,
            proto_normal,
            proto_abnormal,
            resid_abnormal)
