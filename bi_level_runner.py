import argparse
import random

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple
import time

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.stateless import functional_call
from tqdm import tqdm

from model import Model
from scorer import Scorer
from utils import *


LOSS_COMPONENTS: Tuple[str, ...] = (
    'bce', 'margin', 'rec', 'gen', 'div', 'entropy', 'align',
    'resid', 'resid_normal', 'resid_abnormal', 'resid_contrast'
)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    mean = scores.mean()
    std = scores.std()
    if torch.isnan(std) or std < 1e-6:
        return scores - mean
    return (scores - mean) / (std + 1e-6)


def proto_anomaly_scores(residuals: Optional[torch.Tensor],
                         proto_normal: Optional[torch.Tensor],
                         proto_abnormal: Optional[torch.Tensor],
                         beta: float) -> Optional[torch.Tensor]:
    if residuals is None or proto_normal is None or proto_abnormal is None:
        return None
    resid = residuals.squeeze(0)
    proto_n = proto_normal.squeeze(0)
    proto_a = proto_abnormal.squeeze(0)
    if resid.numel() == 0 or proto_n.numel() == 0 or proto_a.numel() == 0:
        return None
    proto_n = proto_n.unsqueeze(0).expand_as(resid)
    proto_a = proto_a.unsqueeze(0).expand_as(resid)
    sim_abn = F.cosine_similarity(resid, proto_a, dim=1)
    sim_norm = F.cosine_similarity(resid, proto_n, dim=1)
    return torch.exp(sim_abn) + beta * torch.exp(-sim_norm)


def blend_logits_with_proto(logits: torch.Tensor,
                            residuals: Optional[torch.Tensor],
                            proto_normal: Optional[torch.Tensor],
                            proto_abnormal: Optional[torch.Tensor],
                            alpha: float,
                            beta: float) -> torch.Tensor:
    base = torch.squeeze(logits)
    alpha = float(alpha)
    beta = float(beta)
    proto_scores = proto_anomaly_scores(residuals, proto_normal, proto_abnormal, beta)
    if proto_scores is None or alpha >= 0.999:
        return base
    if alpha <= 0.001:
        return proto_scores
    base_norm = _normalize_scores(base)
    proto_norm = _normalize_scores(proto_scores)
    return alpha * base_norm + (1.0 - alpha) * proto_norm


def normalize_component_list(raw: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated component names; returns None to keep all."""
    if raw is None:
        return None
    tokens = [tok.strip().lower() for tok in raw.split(',') if tok.strip()]
    if not tokens:
        return None
    if any(tok in ('all', 'full') for tok in tokens):
        return None
    normalized = [tok for tok in tokens if tok in LOSS_COMPONENTS]
    unknown = sorted(set(tokens) - set(normalized))
    if unknown:
        print(f"[Warning] Unknown meta loss components ignored: {','.join(unknown)}")
    return normalized or None



def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bi-level GGAD training")

    parser.add_argument('--dataset', type=str, default='reddit')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--var', type=float, default=0.0)

    parser.add_argument('--inner_steps', type=int, default=1,
                        help='每轮上层更新前，下层执行的步数')
    parser.add_argument('--scorer_hidden', type=int, default=128,
                        help='评分器隐藏层宽度')
    parser.add_argument('--lambda_bce', type=float, default=1.0,
                        help='BCE 损失权重，允许重新平衡监督项')
    parser.add_argument('--lambda_margin', type=float, default=1.0,
                        help='置信度间隔损失权重')
    parser.add_argument('--lambda_rec', type=float, default=2.0,
                        help='重构损失权重')
    parser.add_argument('--lambda_gen', type=float, default=0.5,
                        help='生成器对抗项权重')
    parser.add_argument('--lambda_div', type=float, default=0.1,
                        help='多样性损失权重')
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help='Alignment loss 权重（残差与正常原型的距离）')
    parser.add_argument('--lambda_resid', type=float, default=0.0,
                        help='Residual norm minimization weight')
    parser.add_argument('--lambda_resid_normal', type=float, default=0.0,
                        help='Residual shrinkage weight for labeled normal nodes')
    parser.add_argument('--lambda_resid_abnormal', type=float, default=0.0,
                        help='Residual expansion weight for labeled abnormal nodes')
    parser.add_argument('--lambda_resid_contrast', type=float, default=1.0,
                        help='Residual margin enforcement between normal/anomalous nodes')
    parser.add_argument('--resid_margin', type=float, default=0.2,
                        help='Required gap (anomaly - normal) in residual norms')
    parser.add_argument('--lr_phi', type=float, default=1e-4,
                        help='Scorer 的学习率')
    parser.add_argument('--meta_inner_lr', type=float, default=None,
                        help='上层隐式微分采用的内层学习率（若为空则与 lr 相同）')
    parser.add_argument('--gumbel_tau', type=float, default=1.0,
                        help='Gumbel-Softmax 温度')
    parser.add_argument('--lambda_entropy', type=float, default=1e-3,
                        help='评分器输出的熵正则系数，防止权重塌缩')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='梯度裁剪阈值，0 或负值表示不裁剪')
    parser.add_argument('--logit_clip', type=float, default=10.0,
                        help='对判别器 logits 做裁剪以避免数值爆炸，<=0 表示不裁剪')
    parser.add_argument('--weight_floor', type=float, default=0.05,
                        help='伪异常样本最小权重，<=0 表示不启用')
    parser.add_argument('--meta_loss_components', type=str, default='bce,align',
                        help='指定参与上层优化的损失项，逗号分隔（bce,margin,rec,gen,div,entropy,align），all 表示全部')
    parser.add_argument('--warmup_epochs', type=int, default=50,
                        help='前若干 epoch 仅训练下层，关闭评分器赋权与上层优化')
    parser.add_argument('--proto_score_alpha', type=float, default=0.5,
                        help='融合判别器 logits 与残差原型得分的权重 (1.0 表示仅 logits)')
    parser.add_argument('--proto_beta', type=float, default=4.0,
                        help='AnomalyGFM 风格残差评分中的负原型惩罚系数')

    args = parser.parse_args()

    if args.lr is None:
        args.lr = 1e-3
    if args.meta_inner_lr is None:
        args.meta_inner_lr = args.lr

    if args.dataset in ['photo']:
        args.num_epoch = args.num_epoch or 100
    elif args.dataset in ['elliptic']:
        args.num_epoch = args.num_epoch or 150
    elif args.dataset in ['reddit']:
        args.num_epoch = args.num_epoch or 300
    

    dataset_name = (args.dataset or '').lower()

    if args.dataset in ['reddit', 'photo']:
        args.mean = 0.02
        args.var = 0.01
    else:
        args.mean = 0.0
        args.var = 0.0

    # --- Dataset-specific safety knobs ------------------------------------
   

    if dataset_name == 'reddit':
        # Reddit needs alignment to train the Predictor, otherwise residuals are random noise.
        # But we disable proto-score blending because linear distance is bad for heterophily.
        if args.lambda_align <= 1e-6:
            print('[Auto-Config] Reddit detected: enabling alignment (lambda_align=1.0) to train Predictor.')
            args.lambda_align = 1.0
        
        print('[Auto-Config] Reddit detected: disabling prototype blending (alpha=1.0).')
        args.proto_score_alpha = 1.0

    if args.lambda_align <= 1e-6 and args.proto_score_alpha < 1.0:
        print('[Auto-Fix] lambda_align≈0 -> prototypes untrained. Forcing proto_score_alpha=1.0 to avoid noise.')
        args.proto_score_alpha = 1.0

    args.meta_loss_components = normalize_component_list(args.meta_loss_components)

    return args


def set_seed(seed: int):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def diversity_loss(emb_fake: torch.Tensor) -> torch.Tensor:
    if emb_fake.dim() == 3:
        emb_fake = emb_fake.squeeze(0)
    n = emb_fake.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=emb_fake.device)
    diff = emb_fake.unsqueeze(1) - emb_fake.unsqueeze(0)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8)
    mask = torch.triu(torch.ones_like(dist), diagonal=1)
    num_pairs = mask.sum().clamp(min=1)
    mean_dist = (dist * mask).sum() / num_pairs
    return -mean_dist


def main():
    args = build_args()
    print('Dataset:', args.dataset)
    gpu_id = 2
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)  
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    if torch.cuda.is_available():
        try:
            current = torch.cuda.current_device()
            print(f"Current CUDA device index: {current}")
        except Exception:
            pass
    set_seed(args.seed)

    # Load and preprocess data
    print(f"Loading dataset: {args.dataset}...")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, _, _, normal_label_idx, abnormal_label_idx = load_mat(args.dataset)
    print(f"Successfully loaded {args.dataset}. Nodes: {features.shape[0]}, Features: {features.shape[1]}")
    print(f"Anomaly Label Distribution: {Counter(ano_label)}")
    ano_label = np.array(ano_label)

    if args.dataset in ['reddit', 'elliptic', 'YelpChi']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    identity = sp.eye(adj.shape[0])

    # Keep original sparse adjacency for later affinity computation
    raw_adj = adj.copy().tocsr()

    # Message passing adjacency: add self-loops then symmetric normalize
    adj_with_loop = raw_adj + identity
    adj_prop = normalize_adj(adj_with_loop).todense()

    # Residual adjacency: remove self-loops, row-normalize neighbors only
    adj_no_loop = raw_adj.tolil()
    adj_no_loop.setdiag(0.0)
    adj_no_loop = adj_no_loop.tocsr()
    rowsum = np.array(adj_no_loop.sum(1)).flatten()
    rowsum[rowsum == 0] = 1.0
    d_inv = 1.0 / rowsum
    adj_resid = sp.diags(d_inv).dot(adj_no_loop).todense()

    # Affinity adjacency uses original graph with self-loops
    raw_adj_with_loop = (raw_adj + identity).todense()

    features = torch.FloatTensor(features[np.newaxis]).to(device)
    adj_prop = torch.FloatTensor(adj_prop[np.newaxis]).to(device)
    adj_resid = torch.FloatTensor(adj_resid[np.newaxis]).to(device)
    raw_adj = torch.FloatTensor(raw_adj_with_loop[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    raw_adj_sq = torch.squeeze(raw_adj)

    idx_train_list = [int(i) for i in idx_train]
    idx_val_list = [int(i) for i in idx_val]
    normal_all = [int(i) for i in normal_label_idx]
    abnormal_all = [int(i) for i in abnormal_label_idx]

    def intersect(a, b):
        return sorted(list(set(a).intersection(set(b))))

    train_normal_idx = intersect(normal_all, idx_train_list) or normal_all
    train_abnormal_idx = intersect(abnormal_all, idx_train_list) or abnormal_all
    val_normal_idx = intersect(normal_all, idx_val_list)
    val_abnormal_idx = intersect(abnormal_all, idx_val_list)
    if len(val_normal_idx) == 0 and len(idx_val_list) > 0:
        val_normal_idx = idx_val_list

    def to_tensor(idx_list):
        if len(idx_list) == 0:
            return torch.tensor([], device=device, dtype=torch.long)
        return torch.tensor(idx_list, device=device, dtype=torch.long)

    train_normal_idx_tensor = to_tensor(train_normal_idx)
    train_abnormal_idx_tensor = to_tensor(train_abnormal_idx)
    val_normal_idx_tensor = to_tensor(val_normal_idx)
    val_abnormal_idx_tensor = to_tensor(val_abnormal_idx)

    num_train_normal = len(train_normal_idx)

    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout, dropout=args.drop_prob).to(device)
    # Scorer now takes Residual only -> embedding_dim
    scorer = Scorer(in_dim=args.embedding_dim * 2, hidden_dim=args.scorer_hidden).to(device)

    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params}")
    # --------------------------------

    theta_optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    phi_optim = torch.optim.Adam(scorer.parameters(), lr=args.lr_phi, weight_decay=args.weight_decay)

    pos_weight = torch.tensor([args.negsamp_ratio], device=device)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def build_train_label(num_fake: int) -> torch.Tensor:
        zeros = torch.zeros(num_train_normal, device=device)
        ones = torch.ones(max(num_fake, 0), device=device)
        return torch.cat((zeros, ones), 0).unsqueeze(0).unsqueeze(-1)

    def compute_train_losses(emb: torch.Tensor,
                             logits: torch.Tensor,
                             emb_con: Optional[torch.Tensor],
                             emb_abnormal: Optional[torch.Tensor],
                             resid_all: Optional[torch.Tensor] = None,
                             resid_fake: Optional[torch.Tensor] = None,
                             proto_normal: Optional[torch.Tensor] = None,
                             proto_abnormal: Optional[torch.Tensor] = None,
                             resid_real: Optional[torch.Tensor] = None,
                             detach_weights: bool = True,
                             force_uniform_weights: bool = False,
                             components_subset: Optional[Sequence[str]] = None) -> Tuple[torch.Tensor, dict]:
        logits_for_loss = logits
        if args.logit_clip and args.logit_clip > 0:
            logits_for_loss = torch.clamp(logits, min=-args.logit_clip, max=args.logit_clip)

        num_all = logits_for_loss.size(1)
        num_fake = emb_con.size(0) if emb_con is not None else 0
        num_real = max(num_all - num_train_normal - num_fake, 0)

        label_parts = [torch.zeros(num_train_normal, device=device)]
        if num_fake > 0:
            label_parts.append(torch.ones(num_fake, device=device))
        if num_real > 0:
            label_parts.append(torch.ones(num_real, device=device))
        lbl = torch.cat(label_parts, dim=0).unsqueeze(0).unsqueeze(-1)

        loss_all = b_xent(logits_for_loss, lbl).squeeze(0).squeeze(-1)
        ptr = 0
        loss_norm = loss_all[ptr:ptr + num_train_normal]
        ptr += num_train_normal
        loss_fake = loss_all[ptr:ptr + num_fake] if num_fake > 0 else torch.tensor([], device=device)
        ptr += num_fake
        loss_real = loss_all[ptr:ptr + num_real] if num_real > 0 else torch.tensor([], device=device)

        stats = {}
        weight_mean = torch.tensor(1.0, device=device)
        entropy_term = torch.tensor(0.0, device=device)
        weights = None
        resid_all_sq = resid_all.squeeze(0) if resid_all is not None else None
        if num_fake > 0:
            # Scorer consumes Hybrid (Embedding + Residual)
            if resid_fake is not None:
                # resid_fake is the residual (1, n_fake, D)
                resid_fake_sq = resid_fake.squeeze(0)
                
                # Scorer input: Hybrid
                scorer_input = torch.cat([emb_con, resid_fake_sq], dim=-1)
                
                resid_fake_norm = torch.norm(resid_fake_sq, dim=1).mean()
                stats['resid_fake_norm'] = resid_fake_norm.detach()
                # Minimize residual norm to generate stealthy anomalies (closer to normal manifold)
                loss_resid_norm = resid_fake_norm
            else:
                # Fallback (should not happen with current model)
                emb_abn_sq = emb_abnormal.squeeze(0)
                # Fake residual if not provided (assuming 0 residual for fallback)
                scorer_input = torch.zeros(emb_abn_sq.size(0), emb_abn_sq.size(1) * 2, device=device)
                loss_resid_norm = torch.tensor(0.0, device=device)
                
            if force_uniform_weights:
                weights = torch.ones(num_fake, device=device)
                entropy = torch.tensor(0.0, device=device)
                gumbel_out = torch.stack([weights, weights], dim=-1) / 2
            else:
                logits_sel = scorer(scorer_input)
                gumbel_out = F.gumbel_softmax(logits_sel, tau=args.gumbel_tau, hard=False)
                weights = gumbel_out[:, 0]
                if args.weight_floor and args.weight_floor > 0:
                    weights = torch.clamp(weights, min=args.weight_floor)
                entropy = -torch.sum(gumbel_out * torch.log(gumbel_out + 1e-8), dim=-1).mean()
            entropy_term = -entropy
            if detach_weights:
                weights = weights.detach()
            weight_mean = torch.mean(weights)
            loss_fake = loss_fake * weights
        if weights is None and num_fake > 0:
            weights = torch.ones(num_fake, device=device)
        stats['weight_mean'] = weight_mean.detach()
        if 'loss_resid_norm' not in locals():
            loss_resid_norm = torch.tensor(0.0, device=device)

        # Balanced BCE Loss to handle class imbalance (Normal >> Anomaly)
        loss_terms = []
        loss_norm_mean = loss_norm.mean()
        loss_terms.append(loss_norm_mean)
        loss_fake_mean = loss_fake.mean() if loss_fake.numel() > 0 else torch.tensor(0.0, device=device)
        if loss_fake.numel() > 0:
            loss_terms.append(loss_fake_mean)
        loss_real_mean = loss_real.mean() if loss_real.numel() > 0 else torch.tensor(0.0, device=device)
        if loss_real.numel() > 0:
            loss_terms.append(loss_real_mean)
        loss_bce = torch.mean(torch.stack(loss_terms)) if loss_terms else torch.tensor(0.0, device=device)
        stats['loss_bce'] = loss_bce.detach()

        emb_sq = torch.squeeze(emb)
        emb_inf = torch.norm(emb_sq, dim=-1, keepdim=True)
        emb_inf = torch.pow(emb_inf, -1)
        emb_inf[torch.isinf(emb_inf)] = 0.
        emb_norm = emb_sq * emb_inf
        sim_matrix = torch.mm(emb_norm, emb_norm.T)
        similar_matrix = sim_matrix * raw_adj_sq
        r_inv = torch.pow(torch.sum(raw_adj_sq, 0), -1)
        r_inv[torch.isinf(r_inv)] = 0.
        affinity = torch.sum(similar_matrix, 0) * r_inv
        affinity_normal_mean = torch.mean(affinity[train_normal_idx_tensor]) if train_normal_idx_tensor.numel() > 0 else torch.tensor(0.0, device=device)
        affinity_abnormal_mean = torch.mean(affinity[train_abnormal_idx_tensor]) if train_abnormal_idx_tensor.numel() > 0 else torch.tensor(0.0, device=device)
        confidence_margin = 0.7
        loss_margin = (confidence_margin - (affinity_normal_mean - affinity_abnormal_mean)).clamp_min(min=0)
        stats['loss_margin'] = loss_margin.detach()

        # Prototype-level residual alignment between fake and real anomalies
        loss_rec = torch.tensor(0.0, device=device)
        loss_rec_fake = torch.tensor(0.0, device=device)
        if resid_fake is not None:
            resid_fake_sq = resid_fake.squeeze(0)
            if resid_fake_sq.numel() > 0:
                if proto_abnormal is not None and proto_abnormal.numel() > 0:
                    target = proto_abnormal.squeeze(0).unsqueeze(0).expand_as(resid_fake_sq)
                elif resid_real is not None and resid_real.numel() > 0:
                    target = resid_real.squeeze(0).detach().mean(dim=0, keepdim=True)
                    target = target.expand_as(resid_fake_sq)
                else:
                    target = None
                if target is not None:
                    per_sample = torch.mean((resid_fake_sq - target) ** 2, dim=1)
                    if weights is not None and per_sample.size(0) == weights.size(0):
                        loss_rec_fake = torch.mean(per_sample * weights)
                    else:
                        loss_rec_fake = per_sample.mean()
        loss_rec_real = torch.tensor(0.0, device=device)
        if proto_abnormal is not None and proto_abnormal.numel() > 0 and resid_real is not None and resid_real.numel() > 0:
            resid_real_sq = resid_real.squeeze(0)
            target_real = proto_abnormal.squeeze(0).unsqueeze(0).expand_as(resid_real_sq)
            loss_rec_real = F.mse_loss(resid_real_sq, target_real)
        loss_rec = loss_rec_fake + loss_rec_real
        stats['loss_rec'] = loss_rec.detach()

        logits_fake = logits_for_loss[:, num_train_normal:, :]
        gen_term = torch.tensor(0.0, device=device)
        if logits_fake.numel() > 0:
            # Generator wants logits_fake to be classified as Normal (Label 0)
            # Standard GAN generator loss: minimize BCE(logits_fake, 0)
            # equivalent to maximizing BCE(logits_fake, 1) which is -log(D(G(z)))
            # But we use min log(1 - D(G(z))) here.
            # BCEWithLogitsLoss(logits, 0) = - [0 * log(sig) + 1 * log(1-sig)] = - log(1 - sig)
            gen_term = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
        stats['loss_gen'] = gen_term.detach()

        div_term = torch.tensor(0.0, device=device)
        if args.lambda_div > 0 and emb_abnormal.numel() > 0:
            # div_term = diversity_loss(emb_abnormal)
            div_term = diversity_loss(emb_con)
        stats['loss_div'] = div_term.detach()

        stats['loss_entropy'] = entropy_term.detach()

        # Alignment loss: encourage normal residuals to be close to normal prototype
        align_term = torch.tensor(0.0, device=device)
        if args.lambda_align and args.lambda_align > 0 and resid_all_sq is not None and proto_normal is not None:
            if train_normal_idx_tensor.numel() > 0:
                resid_normals = resid_all_sq[train_normal_idx_tensor]
                target = proto_normal.squeeze(0).unsqueeze(0).expand_as(resid_normals)
                align_term = F.mse_loss(resid_normals, target)
        stats['loss_align'] = align_term.detach()
        stats['loss_resid'] = loss_resid_norm.detach()

        # Residual structure regularization
        resid_normal_term = torch.tensor(0.0, device=device)
        resid_abnormal_term = torch.tensor(0.0, device=device)
        resid_contrast_term = torch.tensor(0.0, device=device)
        resid_normal_mean = torch.tensor(0.0, device=device)
        resid_abnormal_mean = torch.tensor(0.0, device=device)
        if resid_all_sq is not None:
            if train_normal_idx_tensor.numel() > 0:
                resid_normals = torch.norm(resid_all_sq[train_normal_idx_tensor], dim=1)
                resid_normal_mean = resid_normals.mean()
                resid_normal_term = resid_normal_mean
                stats['resid_normal_mean'] = resid_normal_mean.detach()
            if train_abnormal_idx_tensor.numel() > 0:
                resid_abnormals = torch.norm(resid_all_sq[train_abnormal_idx_tensor], dim=1)
                resid_abnormal_mean = resid_abnormals.mean()
                resid_abnormal_term = -resid_abnormal_mean
                stats['resid_abnormal_mean'] = resid_abnormal_mean.detach()
            if train_normal_idx_tensor.numel() > 0 and train_abnormal_idx_tensor.numel() > 0:
                resid_contrast_term = F.relu(args.resid_margin + resid_normal_mean - resid_abnormal_mean)
        stats['loss_resid_normal'] = resid_normal_term.detach()
        stats['loss_resid_abnormal'] = resid_abnormal_term.detach()
        stats['loss_resid_contrast'] = resid_contrast_term.detach()
        if proto_normal is not None and proto_abnormal is not None:
            proto_cos = F.cosine_similarity(proto_normal, proto_abnormal, dim=-1)
            stats['proto_cos'] = proto_cos.detach()

        # Add residual norm minimization to Generator loss
        # Use args.lambda_resid
        
        component_terms = OrderedDict([
            ('bce', args.lambda_bce * loss_bce),
            ('margin', args.lambda_margin * loss_margin),
            ('rec', args.lambda_rec * loss_rec),
            ('gen', args.lambda_gen * gen_term),
            ('div', args.lambda_div * div_term),
            ('entropy', args.lambda_entropy * entropy_term),
            ('align', args.lambda_align * align_term),
            ('resid', args.lambda_resid * loss_resid_norm),
            ('resid_normal', args.lambda_resid_normal * resid_normal_term),
            ('resid_abnormal', args.lambda_resid_abnormal * resid_abnormal_term),
            ('resid_contrast', args.lambda_resid_contrast * resid_contrast_term),
        ])
        total_loss = torch.stack(list(component_terms.values())).sum()
        stats['total_loss'] = total_loss.detach()

        loss_to_optimize = total_loss
        if components_subset:
            selected = [name for name in components_subset if name in component_terms]
            if selected:
                loss_to_optimize = torch.stack([component_terms[name] for name in selected]).sum()

        return loss_to_optimize, stats

    def clone_model_params(module: nn.Module) -> OrderedDict:
        return OrderedDict((name, param.detach().clone().requires_grad_(True))
                           for name, param in module.named_parameters())

    def meta_forward(params_dict: OrderedDict, train_flag: bool):
        return functional_call(model, params_dict,
                   (features, adj_prop, adj_resid,
                train_normal_idx_tensor, train_normal_idx_tensor, train_abnormal_idx_tensor,
                        train_flag, args))

    def compute_validation_loss_from_params(params_dict: OrderedDict):
        seed_idx = val_normal_idx_tensor if val_normal_idx_tensor.numel() > 0 else train_normal_idx_tensor
        _, _, logits_v, _, _, _, _, _, _, _ = functional_call(
            model, params_dict,
            (features, adj_prop, adj_resid,
             seed_idx, val_normal_idx_tensor, val_abnormal_idx_tensor,
             False, args))
        parts = []
        labels_parts = []
        if val_normal_idx_tensor.numel() > 0:
            part = logits_v[:, val_normal_idx_tensor, :]
            parts.append(part)
            labels_parts.append(torch.zeros_like(part))
        if val_abnormal_idx_tensor.numel() > 0:
            part = logits_v[:, val_abnormal_idx_tensor, :]
            parts.append(part)
            labels_parts.append(torch.ones_like(part))
        if not parts:
            return None
        logits_cat = torch.cat(parts, dim=1)
        labels_cat = torch.cat(labels_parts, dim=1)
        return F.binary_cross_entropy_with_logits(logits_cat, labels_cat)

    total_val_nodes = val_normal_idx_tensor.numel() + val_abnormal_idx_tensor.numel()

    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    epoch_start_time = time.time() # 记录每个 epoch 开始时间
            
    train_start_time = time.time()
    # ------------------------------------

    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Bi-level Training')
        last_train_stats = {}
        for epoch in range(args.num_epoch):
            is_warmup = epoch < max(args.warmup_epochs, 0)
            model.train()
            scorer.eval()
            last_weight_mean = 1.0
            train_logits_snapshot = None
            for inner_step in range(args.inner_steps):
                theta_optim.zero_grad()
                emb, _, logits, emb_con, emb_abnormal, resid_all, resid_fake, proto_norm, proto_abn, resid_abn = model(
                    features, adj_prop, adj_resid,
                    train_normal_idx_tensor, train_normal_idx_tensor, train_abnormal_idx_tensor,
                    True, args)
                L_train, stats = compute_train_losses(
                    emb, logits, emb_con, emb_abnormal,
                    resid_all=resid_all, resid_fake=resid_fake,
                    proto_normal=proto_norm,
                    proto_abnormal=proto_abn,
                    resid_real=resid_abn,
                    detach_weights=True,
                    force_uniform_weights=is_warmup)
                L_train.backward()
                if epoch % 10 == 0 and inner_step == 0:
                     print(f"[DEBUG] Grad Norm Gen: {model.gen_mlp[0].weight.grad.norm().item():.4f}")
                     print(f"[DEBUG] Emb Con Norm: {emb_con.norm(dim=1).mean().item():.4f}" if emb_con is not None else "[DEBUG] Emb Con Norm: n/a")
                     if emb_abnormal is not None:
                         print(f"[DEBUG] Emb Abn Norm: {emb_abnormal.norm(dim=-1).mean().item():.4f}")
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                theta_optim.step()
                last_weight_mean = float(stats['weight_mean'].item()) if 'weight_mean' in stats else 1.0
                last_train_stats = {k: v.item() if torch.is_tensor(v) else float(v) for k, v in stats.items()}
                train_logits_snapshot = logits.detach().cpu()

            meta_val_loss = None
            if total_val_nodes > 0 and not is_warmup:
                scorer.train()
                phi_optim.zero_grad()
                params_clone = clone_model_params(model)
                emb_meta, _, logits_meta, emb_con_meta, emb_abnormal_meta, resid_meta_all, resid_meta_fake, proto_meta_norm, proto_meta_abn, resid_meta_abn = meta_forward(params_clone, True)
                L_train_meta, stats_meta = compute_train_losses(
                    emb_meta, logits_meta, emb_con_meta, emb_abnormal_meta,
                    resid_all=resid_meta_all, resid_fake=resid_meta_fake,
                    proto_normal=proto_meta_norm,
                    proto_abnormal=proto_meta_abn,
                    resid_real=resid_meta_abn,
                    detach_weights=False,
                    force_uniform_weights=False,
                    components_subset=args.meta_loss_components)
                grads = torch.autograd.grad(L_train_meta, params_clone.values(), create_graph=True, allow_unused=True)
                updated_params = OrderedDict()
                for (name, param), grad in zip(params_clone.items(), grads):
                    if grad is None:
                        updated_params[name] = param
                    else:
                        updated_params[name] = param - args.meta_inner_lr * grad
                val_loss = compute_validation_loss_from_params(updated_params)
                if val_loss is not None:
                    val_loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_clip)
                    phi_optim.step()
                    meta_val_loss = val_loss.detach().item()
                scorer.eval()

    
            model.eval()
            with torch.no_grad():
                _, _, logits, _, _, resid_all, _, proto_eval_norm, proto_eval_abn, _ = model(
                    features, adj_prop, adj_resid,
                    train_normal_idx_tensor, train_normal_idx_tensor, train_abnormal_idx_tensor,
                    False, args)
                score_tensor = blend_logits_with_proto(logits, resid_all, proto_eval_norm, proto_eval_abn,
                                                       args.proto_score_alpha, args.proto_beta)
                score_np_full = np.squeeze(score_tensor.detach().cpu().numpy())
                idx_val_arr = np.array(idx_val, dtype=np.int64)
                valid_mask = (idx_val_arr >= 0) & (idx_val_arr < logits.size(1))
                idx_val_arr = idx_val_arr[valid_mask]
                if idx_val_arr.size == 0:
                    auc_val = 0.0
                    auprc_val = 0.0
                else:
                    logits_val_np = score_np_full[idx_val_arr]
                    labels_val_np = ano_label[idx_val_arr]
                    
                    # Debug: Logit Statistics
                    val_n_mask = (labels_val_np == 0)
                    val_a_mask = (labels_val_np == 1)
                    mean_logit_n = np.mean(logits_val_np[val_n_mask]) if np.sum(val_n_mask) > 0 else 0.0
                    mean_logit_a = np.mean(logits_val_np[val_a_mask]) if np.sum(val_a_mask) > 0 else 0.0
                    print(f"  [DEBUG] Val Logits: Normal={mean_logit_n:.4f}, Anomaly={mean_logit_a:.4f}, Diff={mean_logit_a - mean_logit_n:.4f}")

                    # Standard logits (Anomaly=1 -> High Score)
                    # Discriminator is trained with Normal=0, Fake=1.
                    # So Logits are Anomaly Scores (Higher = Anomaly).
                    auc_val = roc_auc_score(labels_val_np, logits_val_np)
                    if auc_val < 0.5:
                        auc_val = roc_auc_score(labels_val_np, -logits_val_np)
                        auprc_val = average_precision_score(labels_val_np, -logits_val_np)
                    else:
                         auprc_val = average_precision_score(labels_val_np, logits_val_np)

                    # Auto-flip removed for MLP discriminator
                    # if epoch < 50 and auc_val < 0.5:
                    #     print(f"  [Auto-Flip] Validation AUC {auc_val:.4f} < 0.5. Flipping Discriminator Scale.")
                    #     model.disc.scale.data *= -1
                
                # Debug: Print Residual Norms for Train, Val, Test
                resid_norm = torch.norm(resid_all.squeeze(0), dim=1).detach().cpu().numpy()
                
                def get_mean_norm(indices, label_mask):
                    idx = np.array(indices)
                    idx = idx[idx < len(resid_norm)]
                    if len(idx) == 0: return 0.0
                    lbl = ano_label[idx]
                    mask = (lbl == label_mask)
                    if np.sum(mask) == 0: return 0.0
                    return np.mean(resid_norm[idx[mask]])

                tr_n = get_mean_norm(idx_train, 0)
                tr_a = get_mean_norm(idx_train, 1)
                val_n = get_mean_norm(idx_val, 0)
                val_a = get_mean_norm(idx_val, 1)
                te_n = get_mean_norm(idx_test, 0)
                te_a = get_mean_norm(idx_test, 1)
                
                fake_n = last_train_stats.get('resid_fake_norm', 0.0)
                print(f"Resid Norms | Train: N={tr_n:.4f}, A={tr_a:.4f} | Val: N={val_n:.4f}, A={val_a:.4f} | Test: N={te_n:.4f}, A={te_a:.4f} | Fake={fake_n:.4f}")

            L_val_monitor = meta_val_loss if meta_val_loss is not None else -auc_val

            
            if train_logits_snapshot is None:
                auc_train = 0.0
            else:
                logits_np = np.squeeze(train_logits_snapshot.cpu().numpy())
                num_all = train_logits_snapshot.shape[1]
                lbl_np = np.squeeze(build_train_label(max(num_all - num_train_normal, 0)).detach().cpu().numpy())
                try:
                    auc_train = roc_auc_score(lbl_np, logits_np)
                except ValueError:
                    auc_train = 0.5

            train_loss_total = last_train_stats.get('total_loss', float(L_train.detach().item())) if 'total_loss' in last_train_stats else float(L_train.detach().item())
            
            display_parts = [
                f"Epoch: {epoch:04d}",
                f"train_auc={auc_train:.4f}",
                f"val_auc={auc_val:.4f}",
                f"val_auprc={auprc_val:.4f}",
                f"L_total={train_loss_total:.5f}",
                f"L_val={L_val_monitor:.5f}",
                f"w_mean={last_weight_mean:.3f}"
            ]
            if last_train_stats:
                for key in ['loss_bce', 'loss_margin', 'loss_rec', 'loss_gen', 'loss_div', 'loss_entropy', 'loss_align', 'loss_resid',
                            'loss_resid_normal', 'loss_resid_abnormal', 'loss_resid_contrast', 'proto_cos']:
                    if key in last_train_stats:
                        display_parts.append(f"{key}={last_train_stats[key]:.5f}")
            print(' '.join(display_parts))

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    _, _, logits, _, _, resid_all_test, _, proto_test_norm, proto_test_abn, _ = model(
                        features, adj_prop, adj_resid,
                        train_normal_idx_tensor, normal_all, None,
                        False, args)
                    blended_test = blend_logits_with_proto(logits, resid_all_test, proto_test_norm, proto_test_abn,
                                                           args.proto_score_alpha, args.proto_beta)
                    logits_test_np = np.squeeze(blended_test.detach().cpu().numpy())[idx_test]
                    
                    # Debug: Test Logits
                    labels_test = ano_label[idx_test]
                    test_n_mask = (labels_test == 0)
                    test_a_mask = (labels_test == 1)
                    mean_logit_test_n = np.mean(logits_test_np[test_n_mask]) if np.sum(test_n_mask) > 0 else 0.0
                    mean_logit_test_a = np.mean(logits_test_np[test_a_mask]) if np.sum(test_a_mask) > 0 else 0.0
                    print(f"  [DEBUG] Test Logits: Normal={mean_logit_test_n:.4f}, Anomaly={mean_logit_test_a:.4f}, Diff={mean_logit_test_a - mean_logit_test_n:.4f}")

                    # Standard logits
                    # Discriminator is trained with Normal=0, Fake=1.
                    # So Logits are Anomaly Scores (Higher = Anomaly).
                    auc = roc_auc_score(ano_label[idx_test], logits_test_np)
                    if auc < 0.5:
                        auc = roc_auc_score(ano_label[idx_test], -logits_test_np)
                        print(f"  [Auto-Flip] Detected inverted logits. Using -logits for AUC.")
                        # Also flip logits for AP calculation
                        ap = average_precision_score(ano_label[idx_test], -logits_test_np,
                                                     average='macro', pos_label=1, sample_weight=None)
                    else:
                        ap = average_precision_score(ano_label[idx_test], logits_test_np,
                                                     average='macro', pos_label=1, sample_weight=None)
                    print('Testing {} AUC:{:.4f}'.format(args.dataset, auc))
                    print('Testing AP:', ap)

            pbar.update(1)
    
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    avg_epoch_time = total_train_time / args.num_epoch
    
    print(f"\n=== Resource Usage Stats ===")
    print(f"Total Training Time: {total_train_time:.4f} s")
    print(f"Average Time per Epoch: {avg_epoch_time:.4f} s")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) 
        print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")
    
    
    print("Measuring Inference Time...")
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_start_time = time.time()
    
    with torch.no_grad():
        
        _, _, logits, _, _, resid_all_test, _, proto_test_norm, proto_test_abn, _ = model(
            features, adj_prop, adj_resid,
            train_normal_idx_tensor, normal_all, None,
            False, args)
        blended_test = blend_logits_with_proto(logits, resid_all_test, proto_test_norm, proto_test_abn,
                                               args.proto_score_alpha, args.proto_beta)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_end_time = time.time()
    print(f"Inference Time (Full Graph): {inf_end_time - inf_start_time:.4f} s")
    print(f"============================\n")
    # --------------------------------



if __name__ == '__main__':
    main()
