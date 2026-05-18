import torch
import torch.nn as nn
import torch.nn.functional as F


class Scorer(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        """Attention-based Scorer.
        
        Uses the same Gated Fusion logic as the Discriminator to accurately evaluate
        the quality of generated anomalies based on both Semantic and Structural features.
        """
        super(Scorer, self).__init__()
        
        # in_dim is 2*D
        self.d = int(in_dim / 2)
        
        # Feature extraction
        self.feat_emb = nn.Sequential(nn.Linear(self.d, hidden_dim), nn.ReLU())
        self.feat_resid = nn.Sequential(nn.Linear(self.d, hidden_dim), nn.ReLU())
        
        # Attention
        self.att = nn.Linear(hidden_dim * 2, 2)
        
        # Scorer Head
        self.fc_out = nn.Linear(hidden_dim, 2) # 2 logits for Gumbel-Softmax
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算
        x: (N, 2D) or (B, N, 2D)
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, original_shape[-1])
            
        # Split input
        emb = x[:, :self.d]
        resid = x[:, self.d:]
        
        # Extract features
        h_emb = self.feat_emb(emb)
        h_resid = self.feat_resid(resid)
        
        # Attention
        concat = torch.cat([h_emb, h_resid], dim=1)
        weights = F.softmax(self.att(concat), dim=1)
        
        w_emb = weights[:, 0].unsqueeze(1)
        w_resid = weights[:, 1].unsqueeze(1)
        
        # Fusion
        fused = w_emb * h_emb + w_resid * h_resid
        fused = self.dropout(fused)
        
        # Score
        logits = self.fc_out(fused)
        
        return logits
