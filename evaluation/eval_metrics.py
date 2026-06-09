
from tqdm import tqdm
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

from evaluation.eval_utils import masked_mean, cosine_similarity_matrix, spearman_for_orders, model_forward_for_embedding, \
                                slice_even_odd_batch

def batch_to_structural_context(batch) -> torch.Tensor:
    if isinstance(batch, dict):
        mask = batch.get("mask")
        components = []

        if "node_seq_features" in batch:
            node_feats = batch["node_seq_features"]
            for key in ("scalar_features", "categorical_features", "textual_features"):
                if key in node_feats:
                    components.append(masked_mean(node_feats[key].float(), mask))


        if "edge_seq_features" in batch:
            edge_feats = batch["edge_seq_features"]
            for key in ["struct_features"]:
                if key in edge_feats:
                    components.append(masked_mean(edge_feats[key].float(), mask))

        if len(components) == 0:
            raise ValueError("Cannot build structural context from batch.")
        return torch.cat(components, dim=-1)



def get_batch_weak_labels(batch) -> Optional[torch.Tensor]:
    if isinstance(batch, dict) and "weak_labels" in batch:
        return batch["weak_labels"].long()
    if isinstance(batch, (tuple, list)) and len(batch) >= 12:
        b_mode = batch[-1]
        if isinstance(b_mode, torch.Tensor):
            return b_mode[:, 0].long() if b_mode.ndim > 1 else b_mode.long()
    return None


@dataclass
class BatchStats:
    values: List[float]

    def summary(self):
        if len(self.values) == 0:
            return {"mean": float("nan"), "std": float("nan"), "num_batches": 0}
        arr = np.asarray(self.values, dtype=float)
        return {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "num_batches": int(len(arr)),
        }

def compute_unsupervised_batch_metrics(batch, embeddings) -> Dict[str, float]:
    if embeddings.shape[0] < 3:
        return {"spearman_rho": float("nan"), "recall_at_1": float("nan"), "recall_at_5": float("nan"), "recall_at_10": float("nan")}

    n = embeddings.shape[0]

    if n < 3:
        return {
            "spearman_rho": float("nan"),
            "recall_at_1": float("nan"),
            "recall_at_5": float("nan"),
            "recall_at_10": float("nan"),
        }

    struct_ctx = batch_to_structural_context(batch)
    struct_sim = cosine_similarity_matrix(struct_ctx).cpu().numpy()
    emb_sim = cosine_similarity_matrix(embeddings).cpu().numpy()

    spearmans = []
    recalls_1 = []
    recalls_5 = []
    recalls_10 = []

    max_k = min(10, n - 1)

    for i in range(n):
        struct_order = np.argsort(-struct_sim[i])
        emb_order = np.argsort(-emb_sim[i])

        # Remove self before ranking-based evaluation.
        struct_order = struct_order[struct_order != i]
        emb_order = emb_order[emb_order != i]

        if len(struct_order) == 0 or len(emb_order) == 0:
            continue

        spearmans.append(spearman_for_orders(struct_order, emb_order))

        # Fixed relevant set: top-10 structural neighbors.
        relevant = set(struct_order[:max_k].tolist())
        num_relevant = max(1, len(relevant))

        for k, store in [
            (1, recalls_1),
            (5, recalls_5),
            (10, recalls_10),
        ]:
            kk = min(k, len(emb_order))
            retrieved = set(emb_order[:kk].tolist())

            recall_k = len(relevant & retrieved) / num_relevant
            store.append(recall_k)

    return {
        "spearman_rho": float(np.nanmean(spearmans)) if spearmans else float("nan"),
        "recall_at_1": float(np.nanmean(recalls_1)) if recalls_1 else float("nan"),
        "recall_at_5": float(np.nanmean(recalls_5)) if recalls_5 else float("nan"),
        "recall_at_10": float(np.nanmean(recalls_10)) if recalls_10 else float("nan"),
    }


def compute_semi_supervised_batch_metrics(batch, embeddings, model, model_name: str, dataset) -> Dict[str, float]:
    results = {
        "mean_rank": float("nan"),
        "hit_ratio_at_1": float("nan"),
        "hit_ratio_at_5": float("nan"),
        "silhouette": float("nan"),
        "ari": float("nan"),
        "intra_inter_gap": float("nan"),
    }

    query_batch, db_batch = slice_even_odd_batch(batch, model_name)
    if query_batch is not None and db_batch is not None:
        q_emb = model_forward_for_embedding(model, query_batch)
        d_emb = model_forward_for_embedding(model, db_batch)
        sim = cosine_similarity_matrix(torch.cat([q_emb, d_emb], dim=0))
        B = q_emb.shape[0]
        cross_sim = sim[:B, B:].cpu().numpy()

        ranks = []
        hits1 = []
        hits5 = []

        for i in range(B):
            order = np.argsort(-cross_sim[i])
            rank = int(np.where(order == i)[0][0]) + 1
            ranks.append(rank)
            hits1.append(1.0 if rank <= 1 else 0.0)
            hits5.append(1.0 if rank <= 5 else 0.0)

        results["mean_rank"] = float(np.mean(ranks))
        results["hit_ratio_at_1"] = float(np.mean(hits1))
        results["hit_ratio_at_5"] = float(np.mean(hits5))

    weak_labels = get_batch_weak_labels(batch)
    if weak_labels is not None:
        weak_labels = weak_labels.detach().cpu().numpy()
        emb_np = embeddings.detach().cpu().numpy()
        n_clusters = len(np.unique(weak_labels[weak_labels >= 0]))

        if n_clusters >= 2 and emb_np.shape[0] > n_clusters:
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            clusters = km.fit_predict(emb_np)

            try:
                results["silhouette"] = float(silhouette_score(emb_np, clusters))
            except Exception:
                results["silhouette"] = float("nan")

    return results


def summarize_metric_dicts(metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if len(metric_dicts) == 0:
        return {}

    keys = metric_dicts[0].keys()
    summary = {}
    for key in keys:
        stats = BatchStats([m[key] for m in metric_dicts])
        summary[key] = stats.summary()
    return summary

def evaluate_unsup_and_semi(model, test_loader):
    test_dataset = test_loader.dataset
    unsup_metrics = []
    semi_metrics = []

    for batch in tqdm(test_loader, desc=f"[Evaluating"):
        embeddings = model_forward_for_embedding(model, batch)
        unsup_metrics.append(compute_unsupervised_batch_metrics(batch, embeddings))
        semi_metrics.append(compute_semi_supervised_batch_metrics(batch, embeddings, test_dataset))

    results = {
        "unsupervised": summarize_metric_dicts(unsup_metrics),
        "semi_supervised": summarize_metric_dicts(semi_metrics),
    }
    return results


def _multilabel_batch_metrics(logits, target, threshold=0.5, task_names=None):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    target = target.long()

    per_task = {}
    precisions = []
    recalls = []
    f1s = []

    tp_all = []
    fp_all = []
    tn_all = []
    fn_all = []

    num_tasks = target.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]

    for j in range(num_tasks):
        mask = target[:, j] != -1
        task_name = task_names[j]

        if mask.sum() == 0:
            per_task[task_name] = {
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }
            continue

        y_true = target[mask, j]
        y_pred = preds[mask, j]

        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        tn = ((y_true == 0) & (y_pred == 0)).sum().item()
        fn = ((y_true == 1) & (y_pred == 0)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        per_task[task_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        tp_all.append(tp)
        fp_all.append(fp)
        tn_all.append(tn)
        fn_all.append(fn)

    macro = {
        "precision": float(np.mean(precisions)) if len(precisions) > 0 else np.nan,
        "recall": float(np.mean(recalls)) if len(recalls) > 0 else np.nan,
        "f1": float(np.mean(f1s)) if len(f1s) > 0 else np.nan,
        "tp": int(np.sum(tp_all)) if len(tp_all) > 0 else 0,
        "fp": int(np.sum(fp_all)) if len(fp_all) > 0 else 0,
        "tn": int(np.sum(tn_all)) if len(tn_all) > 0 else 0,
        "fn": int(np.sum(fn_all)) if len(fn_all) > 0 else 0,
    }

    return {
        "macro": macro,
        "per_task": per_task,
    }



def _regression_batch_metrics(pred, target, task_names=None):
    num_tasks = target.shape[1]
    if task_names is None:
        task_names = [f"task_{i}" for i in range(num_tasks)]

    per_task = {}

    maes, smapes, rmses = [], [], []
    r2s, spearmans, nrmses = [], [], []

    for j in range(num_tasks):
        mask = target[:, j] != -1
        task_name = task_names[j]

        if mask.sum() == 0:
            per_task[task_name] = {
                "mae": np.nan,
                "smape": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
                "nrmse": np.nan,
            }
            continue

        y_true = target[mask, j]
        y_pred = pred[mask, j]

        diff = y_pred - y_true

        # --- Basic metrics ---
        mae = torch.mean(torch.abs(diff)).item()
        rmse = torch.sqrt(torch.mean(diff ** 2)).item()

        smape = torch.mean(
            2.0 * torch.abs(diff) /
            torch.clamp(torch.abs(y_pred) + torch.abs(y_true), min=1e-8)
        ).item()

        # --- R² ---
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        ss_res = torch.sum(diff ** 2)

        r2 = (1 - ss_res / torch.clamp(ss_tot, min=1e-8)).item()

        # --- Spearman ---
        try:
            spearman_corr = spearmanr(
                y_true.detach().cpu().numpy(),
                y_pred.detach().cpu().numpy()
            ).correlation
        except Exception:
            spearman_corr = np.nan

        # --- NRMSE (std normalized) ---
        std = torch.std(y_true)
        nrmse = (rmse / (std.item() + 1e-8)) if std > 0 else np.nan

        # Store
        per_task[task_name] = {
            "mae": float(mae),
            "smape": float(smape),
            "rmse": float(rmse),
            "r2": float(r2),
            "spearman": float(spearman_corr) if spearman_corr is not None else np.nan,
            "nrmse": float(nrmse),
        }

        maes.append(mae)
        smapes.append(smape)
        rmses.append(rmse)
        r2s.append(r2)
        spearmans.append(spearman_corr)
        nrmses.append(nrmse)

    macro = {
        "mae": float(np.mean(maes)) if maes else np.nan,
        "smape": float(np.mean(smapes)) if smapes else np.nan,
        "rmse": float(np.mean(rmses)) if rmses else np.nan,
        "r2": float(np.nanmean(r2s)) if r2s else np.nan,
        "spearman": float(np.nanmean(spearmans)) if spearmans else np.nan,
        "nrmse": float(np.nanmean(nrmses)) if nrmses else np.nan,
    }

    return {
        "macro": macro,
        "per_task": per_task,
    }
