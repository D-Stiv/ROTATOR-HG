import torch
from tqdm import tqdm


from pretraining.utils import evaluate_embeddings, _is_valid_number, _safe_mean



def compute_eval_score(args, metrics):
    print(f"Computing evaluation score with metrics...")
    eval_metrics_weights_retrieval = eval(args.eval_metrics_weights_retrieval)
    eval_metrics_weights_ranking = eval(args.eval_metrics_weights_ranking)
    eval_metrics_weights_clustering = eval(args.eval_metrics_weights_clustering)
    eval_metrics_weights_global = eval(args.eval_metrics_weights_global)

    metric_weights = {
        "retrieval": {
            "recall@k": eval_metrics_weights_retrieval.get("recall@k", 0.4),
            "trustworthiness": eval_metrics_weights_retrieval.get("trustworthiness", 0.2),
        },
        "ranking": {
            "spearman": eval_metrics_weights_ranking.get("spearman", 1.0),
        },
        "clustering": {
            "silhouette": eval_metrics_weights_clustering.get("silhouette", 0.4),
            "ARI": eval_metrics_weights_clustering.get("ARI", 0.3),
            "Gap": eval_metrics_weights_clustering.get("Gap", 0.3),
        },
        "global": {
            "retrieval": eval_metrics_weights_global.get("retrieval", 0.4),
            "ranking": eval_metrics_weights_global.get("ranking", 0.3),
            "clustering": eval_metrics_weights_global.get("clustering", 0.3),
        },
    }

    def weighted_average(pairs, default=None):
        valid = [(v, w) for v, w in pairs if _is_valid_number(v) and w > 0]
        if len(valid) == 0:
            return default
        num = sum(float(v) * float(w) for v, w in valid)
        den = sum(float(w) for _, w in valid)
        return num / den if den > 0 else default

    recall_keys = [k for k in metrics.keys() if k.startswith("recall@")]
    recall_score = _safe_mean([metrics.get(k) for k in recall_keys], default=None)

    retrieval_score = weighted_average(
        [
            (recall_score, metric_weights["retrieval"]["recall@k"]),
            (metrics.get("trustworthiness"), metric_weights["retrieval"]["trustworthiness"]),
        ],
        default=None,
    )

    ranking_score = weighted_average(
        [
            (metrics.get("spearman"), metric_weights["ranking"]["spearman"]),
        ],
        default=None,
    )

    clustering_score = weighted_average(
        [
            (metrics.get("silhouette"), metric_weights["clustering"]["silhouette"]),
            (metrics.get("ARI"), metric_weights["clustering"]["ARI"]),
            (metrics.get("intra_inter_gap"), metric_weights["clustering"]["Gap"]),
        ],
        default=None,
    )

    eval_score = weighted_average(
        [
            (retrieval_score, metric_weights["global"]["retrieval"]),
            (ranking_score, metric_weights["global"]["ranking"]),
            (clustering_score, metric_weights["global"]["clustering"]),
        ],
        default=float("-inf"),
    )

    return {
        "retrieval_score": retrieval_score,
        "ranking_score": ranking_score,
        "clustering_score": clustering_score,
        "score": eval_score,
    }


def get_metrics(model, dataloader, debug=False):
    model.eval()
    all_route_embs = []
    struct_feats = []
    weak_labels = []
    with torch.no_grad():
        for iter, batch in enumerate(tqdm(dataloader, desc="[Get Emb, Feats amd weak labels for metrics computation")):
            if debug and iter >= 2:
                break
            out = model(batch)
            batch_data = batch
            route_emb = out["route_emb"]  # [B, d_emb]
            all_route_embs.append(route_emb.cpu())

            struct_feat = batch_data['edge_seq_features']['struct_features'].to(model.device) # [B, N, d_struct]
            struct_feat = struct_feat.mean(dim=1)  # [B, d_struct]
            struct_feats.append(struct_feat.cpu())

            if isinstance(batch_data['weak_labels'], torch.Tensor):
                weak_labels.append(batch_data['weak_labels'].cpu())
            else:
                weak_labels.append(torch.tensor(batch_data['weak_labels']))

    all_route_embs = torch.cat(all_route_embs, dim=0)
    struct_feats = torch.cat(struct_feats, dim=0)
    weak_labels = torch.cat(weak_labels, dim=0)

    metrics = evaluate_embeddings(all_route_embs, X_struct=struct_feats, y=weak_labels, K=model.num_clusters)
    return metrics


def compute_eval_score(args, metrics):
    print(f"Computing evaluation score with metrics...")
    eval_metrics_weights_retrieval = eval(args.eval_metrics_weights_retrieval)
    eval_metrics_weights_ranking = eval(args.eval_metrics_weights_ranking)
    eval_metrics_weights_clustering = eval(args.eval_metrics_weights_clustering)
    eval_metrics_weights_global = eval(args.eval_metrics_weights_global)

    metric_weights = {
        "retrieval": {
            "recall@k": eval_metrics_weights_retrieval.get("recall@k", 0.4),
            "trustworthiness": eval_metrics_weights_retrieval.get("trustworthiness", 0.2),
        },
        "ranking": {
            "spearman": eval_metrics_weights_ranking.get("spearman", 1.0),
        },
        "clustering": {
            "silhouette": eval_metrics_weights_clustering.get("silhouette", 0.4),
            "ARI": eval_metrics_weights_clustering.get("ARI", 0.3),
            "Gap": eval_metrics_weights_clustering.get("Gap", 0.3),
        },
        "global": {
            "retrieval": eval_metrics_weights_global.get("retrieval", 0.4),
            "ranking": eval_metrics_weights_global.get("ranking", 0.3),
            "clustering": eval_metrics_weights_global.get("clustering", 0.3),
        },
    }

    def weighted_average(pairs, default=None):
        valid = [(v, w) for v, w in pairs if _is_valid_number(v) and w > 0]
        if len(valid) == 0:
            return default
        num = sum(float(v) * float(w) for v, w in valid)
        den = sum(float(w) for _, w in valid)
        return num / den if den > 0 else default

    recall_keys = [k for k in metrics.keys() if k.startswith("recall@")]
    recall_score = _safe_mean([metrics.get(k) for k in recall_keys], default=None)

    retrieval_score = weighted_average(
        [
            (recall_score, metric_weights["retrieval"]["recall@k"]),
            (metrics.get("trustworthiness"), metric_weights["retrieval"]["trustworthiness"]),
        ],
        default=None,
    )

    ranking_score = weighted_average(
        [
            (metrics.get("spearman"), metric_weights["ranking"]["spearman"]),
        ],
        default=None,
    )

    clustering_score = weighted_average(
        [
            (metrics.get("silhouette"), metric_weights["clustering"]["silhouette"]),
            (metrics.get("ARI"), metric_weights["clustering"]["ARI"]),
            (metrics.get("intra_inter_gap"), metric_weights["clustering"]["Gap"]),
        ],
        default=None,
    )

    eval_score = weighted_average(
        [
            (retrieval_score, metric_weights["global"]["retrieval"]),
            (ranking_score, metric_weights["global"]["ranking"]),
            (clustering_score, metric_weights["global"]["clustering"]),
        ],
        default=float("-inf"),
    )

    return {
        "retrieval_score": retrieval_score,
        "ranking_score": ranking_score,
        "clustering_score": clustering_score,
        "score": eval_score,
    }
