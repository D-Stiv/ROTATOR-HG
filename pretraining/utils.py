import numpy as np
import torch
import math
import torch.nn.functional as F
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), p=2, dim=-1)
    return x @ x.T


def topk_mask(similarity: torch.Tensor, k: int) -> torch.Tensor:
    n = similarity.size(0)
    if n == 0:
        return torch.zeros_like(similarity)

    k = max(0, min(int(k), n - 1))
    mask = torch.zeros_like(similarity)
    if k == 0:
        return mask

    scores = similarity.masked_fill(
        torch.eye(n, dtype=torch.bool, device=similarity.device),
        -1e9,
    )
    idx = torch.topk(scores, k=k, dim=1).indices
    mask.scatter_(1, idx, 1.0)
    return mask


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MaskingStrategy:
    """Masking strategies for trajectory augmentation - Section III.B"""
    
    @staticmethod
    def random_masking(seq, mask_ratio):
        """Random masking (RM) - randomly drop tokens"""
        mask = torch.rand(seq.shape[0], seq.shape[1], device=seq.device) > mask_ratio
        return seq * mask.unsqueeze(-1).float()
    
    @staticmethod
    def consecutive_masking(seq, mask_ratio):
        """Consecutive masking (CM) - mask consecutive points"""
        batch_size, seq_len, _ = seq.shape
        mask_len = int(seq_len * mask_ratio)
        
        mask = torch.ones(batch_size, seq_len, 1, device=seq.device)
        start_idx = torch.randint(0, seq_len - mask_len, (batch_size,), device=seq.device)
        
        for b in range(batch_size):
            mask[b, start_idx[b]:start_idx[b] + mask_len] = 0
        
        return seq * mask
    
    @staticmethod
    def truncation_masking(seq, mask_ratio):
        """Truncation masking (TC) - mask from origin or destination"""
        batch_size, seq_len, _ = seq.shape
        mask_len = int(seq_len * mask_ratio)
        
        mask = torch.ones(batch_size, seq_len, 1, device=seq.device)
        # Randomly choose origin (0) or destination (1)
        mask_from_start = torch.randint(0, 2, (batch_size,), device=seq.device).bool()
        
        for b in range(batch_size):
            if mask_from_start[b]:
                mask[b, :mask_len] = 0
            else:
                mask[b, -mask_len:] = 0
        
        return seq * mask
    
    @staticmethod
    def apply_masking_stack(seq, strategies, mask_ratios):
        """Apply multiple masking strategies - Section III.B"""
        masked_seq = seq.clone()
        for strategy, ratio in zip(strategies, mask_ratios):
            if strategy == 'rm':
                masked_seq = MaskingStrategy.random_masking(masked_seq, ratio)
            elif strategy == 'cm':
                masked_seq = MaskingStrategy.consecutive_masking(masked_seq, ratio)
            elif strategy == 'tc':
                masked_seq = MaskingStrategy.truncation_masking(masked_seq, ratio)
        return masked_seq



def node_sequence_loss(node_updated, node_target, mask):
    pred = node_updated[mask]
    target = node_target[mask]
    return F.mse_loss(pred, target)

def edge_sequence_loss(edge_updated, edge_target, mask):
    pred = edge_updated[mask]
    target = edge_target[mask]
    return F.mse_loss(pred, target)


def route_contrastive_loss(z1, z2, temperature=0.1):
    B = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits_12 = torch.mm(z1, z2.t()) / temperature
    logits_21 = torch.mm(z2, z1.t()) / temperature

    labels = torch.arange(B, device=z1.device)

    return 0.5 * (
        F.cross_entropy(logits_12, labels)
        + F.cross_entropy(logits_21, labels)
    )
    

def neighborhood_contrastive_loss(route_emb, target_features, k=5, temperature=0.1):
    """
    route_emb: [B, D]
    target_features: [B, d_struct] or [B, d_sem]
    """
    B = route_emb.size(0)

    # Similarities
    S_target = cosine_similarity_matrix(target_features)   # original space
    S_emb = cosine_similarity_matrix(route_emb)             # embedding space

    # Get positive mask from original space
    pos_mask = topk_mask(S_target, k=k)

    # Remove self similarity
    S_emb = S_emb / temperature
    S_emb = S_emb - torch.eye(B, device=route_emb.device) * 1e9

    exp_sim = torch.exp(S_emb)

    # Numerator: positives
    numerator = (exp_sim * pos_mask).sum(dim=1)

    # Denominator: all except self
    denominator = exp_sim.sum(dim=1)

    loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8))

    return loss.mean()


def pos_neg_neighborhood_contrastive_loss(
    route_emb,
    target_features,
    pos_k=5,
    neg_k=20,
    temperature=0.1,
    margin=None,
):
    """
    Pull top-k similar routes together and push bottom-k dissimilar routes apart.
    Middle-similarity routes are ignored.
    """

    # chek positive and negative k base on batch size
    B = route_emb.size(0)
    device = route_emb.device
    if B < 2:
        return route_emb.sum() * 0.0

    pos_k = max(1, min(int(pos_k), B - 1))
    neg_k = max(1, min(int(neg_k), B - 1))

    S_target = cosine_similarity_matrix(target_features)
    S_emb = cosine_similarity_matrix(route_emb)

    eye = torch.eye(B, device=device).bool()
    S_target = S_target.masked_fill(eye, -1e9)

    pos_idx = torch.topk(S_target, k=pos_k, dim=1).indices
    neg_idx = torch.topk(-S_target, k=neg_k, dim=1).indices

    pos_mask = torch.zeros(B, B, device=device)
    neg_mask = torch.zeros(B, B, device=device)

    pos_mask.scatter_(1, pos_idx, 1.0)
    neg_mask.scatter_(1, neg_idx, 1.0)

    logits = S_emb / temperature
    logits = logits.masked_fill(eye, -1e9)

    exp_logits = torch.exp(logits)

    pos_score = (exp_logits * pos_mask).sum(dim=1)
    neg_score = (exp_logits * neg_mask).sum(dim=1)

    loss = -torch.log((pos_score + 1e-8) / (pos_score + neg_score + 1e-8))

    if margin is not None:
        pos_sim = (S_emb * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        neg_sim = (S_emb * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)
        margin_loss = F.relu(neg_sim - pos_sim + margin)
        loss = loss + margin_loss

    return loss.mean()


def global_similarity_kl_loss(route_emb, struct_feat, sem_feat, temperature=0.1):
    """
    Preserves global similarity geometry
    """
    S_struct = cosine_similarity_matrix(struct_feat)
    S_sem = cosine_similarity_matrix(sem_feat)

    S_target = 0.5 * (S_struct + S_sem)
    S_emb = cosine_similarity_matrix(route_emb)

    # Remove diagonal
    B = route_emb.size(0)
    mask = torch.eye(B, device=route_emb.device).bool()

    S_target = S_target.masked_fill(mask, -1e9)
    S_emb = S_emb.masked_fill(mask, -1e9)

    # Convert to distributions
    P_target = F.softmax(S_target / temperature, dim=1)
    P_emb = F.log_softmax(S_emb / temperature, dim=1)

    # KL divergence
    loss = F.kl_div(P_emb, P_target, reduction="batchmean")

    return loss

def soft_clustering_loss(route_emb, weak_labels, cluster_centers, temperature=0.5, dist_metric='cosine'):
    """
    route_emb: [B, D]
    weak_labels: [B] (LongTensor)
    cluster_centers: [K, D]
    """
    if dist_metric == 'cosine':
        route_emb = F.normalize(route_emb, dim=1)
        cluster_centers = F.normalize(cluster_centers, dim=1)
        dist = torch.matmul(route_emb, cluster_centers.t())  # (B, K)
    elif dist_metric == 'euclidean':
        dist = -torch.cdist(route_emb, cluster_centers, p=2) # [B, K]

    logits = dist / temperature

    loss = F.cross_entropy(
        logits,
        weak_labels,
        ignore_index=-1
    )

    return loss


def transport_consistency_loss(
    node_updated,
    edge_updated,
    edge_mode,
    node_feasible_modes,
    mode_predictor_nodes,
    mode_predictor_edges
):
    """
    edge_mode: (B, L, M)
    node_feasible_modes: (B, L, M)
    """

    edge_logits = mode_predictor_edges(edge_updated)
    node_logits = mode_predictor_nodes(node_updated)

    edge_loss = F.binary_cross_entropy_with_logits(
        edge_logits, edge_mode.float()
    )

    node_loss = F.binary_cross_entropy_with_logits(
        node_logits, node_feasible_modes.float()
    )

    return edge_loss + node_loss


def route_order_contrastive_loss(model, node_seq, cell_seq, edge_seq, edge_ids, mask_valid, temperature=0.1):
    B, L, _ = edge_seq.shape
    device = edge_seq.device

    z_real, *_ = model.get_route_embedding(node_seq, cell_seq, edge_seq, edge_ids, mask_valid)

    shuffled_edge_seq = edge_seq.clone()
    shuffled_node_seq = node_seq.clone()
    shuffled_cell_seq = cell_seq.clone()
    shuffled_edge_ids = edge_ids.clone()

    for b in range(B):
        valid_idx = torch.where(mask_valid[b])[0]
        if valid_idx.numel() > 1:
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
            shuffled_edge_seq[b, valid_idx] = edge_seq[b, perm]
            shuffled_node_seq[b, valid_idx] = node_seq[b, perm]
            shuffled_cell_seq[b, valid_idx] = cell_seq[b, perm]
            shuffled_edge_ids[b, valid_idx] = edge_ids[b, perm]

    z_shuffle, *_ = model.get_route_embedding(
        shuffled_node_seq,
        shuffled_cell_seq,
        shuffled_edge_seq,
        shuffled_edge_ids,
        mask_valid,
    )

    z_real = F.normalize(z_real, dim=-1)
    z_shuffle = F.normalize(z_shuffle, dim=-1)

    logits = torch.mm(z_real, z_real.t()) / temperature
    neg_logits = torch.sum(z_real * z_shuffle, dim=-1, keepdim=True) / temperature

    labels = torch.arange(B, device=device)
    logits = torch.cat([logits, neg_logits], dim=1)

    return F.cross_entropy(logits, labels)



def recall_at_k_fast(Z, X_target, k=5):
    Z = torch.nn.functional.normalize(Z, dim=1)
    X_target = torch.nn.functional.normalize(X_target, dim=1)

    sim_Z = Z @ Z.T
    sim_X = X_target @ X_target.T

    N = Z.size(0)

    # Exclude self by masking diagonal
    mask = torch.eye(N, device=Z.device).bool()
    sim_Z = sim_Z.masked_fill(mask, -1e9)
    sim_X = sim_X.masked_fill(mask, -1e9)

    topk_Z = torch.topk(sim_Z, k, dim=1).indices
    topk_X = torch.topk(sim_X, k, dim=1).indices

    recall = []

    for i in range(N):
        set_Z = set(topk_Z[i].tolist())
        set_X = set(topk_X[i].tolist())

        recall.append(len(set_Z & set_X) / k)

    return sum(recall) / N


def evaluate_embeddings(Z, X_struct=None, y=None, K=None):
    print("Computing evaluation metrics...")
    results = {}

    if Z is None or len(Z) < 2:
        return {
            "trustworthiness": None,
            "spearman": None,
            "recall@1": None,
            "recall@5": None,
            "recall@10": None,
            "silhouette": None,
            "ARI": None,
            "intra_inter_gap": None,
        }

    if not isinstance(Z, torch.Tensor):
        Z = torch.as_tensor(Z)

    if X_struct is not None and not isinstance(X_struct, torch.Tensor):
        X_struct = torch.as_tensor(X_struct)

    if y is not None and not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)

    n_samples = int(Z.size(0))

    # --------------------------------------------------
    # Trustworthiness
    # --------------------------------------------------
    if X_struct is not None and n_samples >= 3:
        try:
            n_neighbors = min(10, max(1, n_samples // 2 - 1))
            if n_neighbors >= 1:
                trust = trustworthiness(
                    X_struct.detach().cpu().numpy(),
                    Z.detach().cpu().numpy(),
                    n_neighbors=n_neighbors,
                )
                results["trustworthiness"] = _safe_float(trust, None)
            else:
                results["trustworthiness"] = None
        except Exception:
            results["trustworthiness"] = None
    else:
        results["trustworthiness"] = None

    # --------------------------------------------------
    # Similarity alignment
    # --------------------------------------------------
    if X_struct is not None and X_struct.size(0) == Z.size(0):
        try:
            S_emb = cosine_similarity_matrix(Z).detach().cpu().numpy()
            S_target = cosine_similarity_matrix(X_struct).detach().cpu().numpy()

            corr = spearmanr(S_emb.reshape(-1), S_target.reshape(-1)).correlation
            results["spearman"] = _safe_float(corr, None)
        except Exception:
            results["spearman"] = None
    else:
        results["spearman"] = None

    # --------------------------------------------------
    # Recall@K
    # --------------------------------------------------
    if X_struct is not None and X_struct.size(0) == Z.size(0) and n_samples >= 2:
        try:
            results["recall@1"] = _safe_float(recall_at_k_fast(Z, X_struct, k=1), None)            
        except Exception:
            results["recall@1"] = None

        try:
            results["recall@5"] = _safe_float(recall_at_k_fast(Z, X_struct, k=5), None)            
        except Exception:
            results["recall@5"] = None

        try:
            results["recall@10"] = _safe_float(recall_at_k_fast(Z, X_struct, k=10), None)
        except Exception:
            results["recall@10"] = None
    else:
        results["recall@1"] = None
        results["recall@5"] = None
        results["recall@10"] = None

    # --------------------------------------------------
    # Clustering
    # --------------------------------------------------
    y_np = None
    n_labels = 0

    if y is not None:
        try:
            y_np = y.detach().cpu().numpy()
            valid_y = y_np[y_np >= 0] if np.issubdtype(y_np.dtype, np.number) else y_np
            n_labels = len(set(valid_y.tolist())) if len(valid_y) > 0 else 0
        except Exception:
            y_np = None
            n_labels = 0

    if K is None:
        if n_labels >= 2:
            K_eff = n_labels
        else:
            K_eff = max(2, min(10, n_samples // 2))
    else:
        K_eff = max(2, min(int(K), max(2, n_samples // 2)))

    if n_samples >= 3 and K_eff < n_samples:
        try:
            z_np = Z.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=K_eff, n_init=10, random_state=0).fit(z_np)

            # silhouette requires at least 2 clusters and fewer than n_samples clusters
            if len(set(kmeans.labels_)) >= 2 and len(set(kmeans.labels_)) < n_samples:
                results["silhouette"] = _safe_float(
                    silhouette_score(z_np, kmeans.labels_),
                    None,
                )
            else:
                results["silhouette"] = None

            if y_np is not None and n_labels >= 2 and len(y_np) == len(kmeans.labels_):
                try:
                    results["ARI"] = _safe_float(
                        adjusted_rand_score(y_np, kmeans.labels_),
                        None,
                    )
                except Exception:
                    results["ARI"] = None
            else:
                results["ARI"] = None

        except Exception:
            results["silhouette"] = None
            results["ARI"] = None
            kmeans = None
    else:
        results["silhouette"] = None
        results["ARI"] = None
        kmeans = None

    # --------------------------------------------------
    # Intra vs Inter gap
    # --------------------------------------------------
    if y is not None and n_labels >= 2 and n_samples >= 2:
        try:
            sim = cosine_similarity_matrix(Z)

            same_mask = y.unsqueeze(1) == y.unsqueeze(0)
            diff_mask = y.unsqueeze(1) != y.unsqueeze(0)

            eye_mask = ~torch.eye(n_samples, dtype=torch.bool, device=sim.device)
            same_mask = same_mask & eye_mask
            diff_mask = diff_mask & eye_mask

            pos_sim = sim[same_mask]
            neg_sim = sim[diff_mask]

            if pos_sim.numel() > 0 and neg_sim.numel() > 0:
                gap = pos_sim.mean() - neg_sim.mean()
                results["intra_inter_gap"] = _safe_float(gap.item(), None)
            else:
                results["intra_inter_gap"] = None
        except Exception:
            results["intra_inter_gap"] = None
    else:
        results["intra_inter_gap"] = None

    return results


def _safe_float(x, default=None):
    return float(x) if _is_valid_number(x) else default


def _safe_mean(values, default=None):
    valid = [float(v) for v in values if _is_valid_number(v)]
    if len(valid) == 0:
        return default
    return sum(valid) / len(valid)

def _is_valid_number(x):
    if x is None:
        return False
    try:
        x = float(x)
    except Exception:
        return False
    return not (math.isnan(x) or math.isinf(x))

