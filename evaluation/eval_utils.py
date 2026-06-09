import os
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Optional, Sequence
import torch.nn as nn
from dataclasses import dataclass



from pretraining.models import InterRoute2Vec



def build_model(args):
    device = args.device

    args_path = "Insert: path to the saved args during pretraining (should be in the same directory as the checkpoint)"

    with open(args_path, "rb") as f:
        saved_args = pickle.load(f)
        saved_args.device = device

    model = InterRoute2Vec(args=saved_args)

    checkpoint_path = "Insert: path to the saved checkpoint for the pretrian model (should be in the same directory as the saved args)"


    state = torch.load(checkpoint_path, map_location=device)

    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def unpack_embedding_output(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict):
        if "z" in output:
            return output["z"]
        if "embedding" in output:
            return output["embedding"]
        raise ValueError(f"Unsupported dict output keys: {list(output.keys())}")

    if isinstance(output, tuple):
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
        raise ValueError("Unsupported tuple output format.")

    raise ValueError(f"Unsupported model output type: {type(output)}")


def model_forward_for_embedding(model, batch):
    with torch.no_grad():
        try:
            output = model(batch, eval_mode=True)
        except TypeError:
            output = model(batch)
    return unpack_embedding_output(output)


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), p=2, dim=-1)
    return x @ x.T


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom


def rank_positions(order: Sequence[int]) -> np.ndarray:
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order, dtype=np.int64)] = np.arange(len(order))
    return pos


def spearman_for_orders(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) <= 1:
        return float("nan")
    ra = rank_positions(a).astype(np.float64)
    rb = rank_positions(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def build_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)



def slice_even_odd_batch(batch, model_name: str):
    if not isinstance(batch, dict):
        return None, None
    def make_copy(source):
        new_batch = {}
        for k, v in source.items():
            if isinstance(v, dict):
                new_batch[k] = {kk: vv for kk, vv in v.items()}
            else:
                new_batch[k] = v
        return new_batch

    
    if "node_seq_even" in batch and "node_seq_odd" in batch:
        q = make_copy(batch)
        d = make_copy(batch)

        q["node_seq_emb"] = batch["node_seq_even"]
        d["node_seq_emb"] = batch["node_seq_odd"]

        if "edge_ids" in batch:
            q["edge_ids"] = batch["edge_ids"][:, ::2]
            d["edge_ids"] = batch["edge_ids"][:, 1::2]

        if "cell_seq_even" in batch:
            q["cell_seq_emb"] = batch["cell_seq_even"]
            d["cell_seq_emb"] = batch["cell_seq_odd"]
        if "edge_seq_even" in batch:
            q["edge_seq_emb"] = batch["edge_seq_even"]
            d["edge_seq_emb"] = batch["edge_seq_odd"]
        if "time_seq_even" in batch:
            q["time_seq"] = batch["time_seq_even"]
            d["time_seq"] = batch["time_seq_odd"]

        if "even_len" in batch:
            q["seq_len"] = batch["even_len"]
            q["mask"] = build_mask_from_lengths(batch["even_len"], q["node_seq_emb"].shape[1])
        if "odd_len" in batch:
            d["seq_len"] = batch["odd_len"]
            d["mask"] = build_mask_from_lengths(batch["odd_len"], d["node_seq_emb"].shape[1])

        return q, d
    
    q = make_copy(batch)
    d = make_copy(batch)

    if "node_seq_emb" in batch:
        q["node_seq_emb"] = batch["node_seq_emb"][:, ::2]
        d["node_seq_emb"] = batch["node_seq_emb"][:, 1::2]
    if "cell_seq_emb" in batch:
        q["cell_seq_emb"] = batch["cell_seq_emb"][:, ::2]
        d["cell_seq_emb"] = batch["cell_seq_emb"][:, 1::2]
    if "edge_seq_emb" in batch:
        q["edge_seq_emb"] = batch["edge_seq_emb"][:, ::2]
        d["edge_seq_emb"] = batch["edge_seq_emb"][:, 1::2]
    if "time_seq" in batch:
        q["time_seq"] = batch["time_seq"][:, ::2]
        d["time_seq"] = batch["time_seq"][:, 1::2]
    if "edge_ids" in batch:
        q["edge_ids"] = batch["edge_ids"][:, ::2]
        d["edge_ids"] = batch["edge_ids"][:, 1::2]

    if "seq_len" in batch:
        even_len = torch.div(batch["seq_len"] + 1, 2, rounding_mode="floor")
        odd_len = torch.div(batch["seq_len"], 2, rounding_mode="floor")
        q["seq_len"] = even_len
        d["seq_len"] = odd_len
        q["mask"] = build_mask_from_lengths(even_len, q["node_seq_emb"].shape[1])
        d["mask"] = build_mask_from_lengths(odd_len, d["node_seq_emb"].shape[1])

    return q, d


def _safe_mean_std(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))



def collect_embeddings(model, loader) -> torch.Tensor:
    chunks = []
    for batch in loader:
        chunks.append(model_forward_for_embedding(model, batch).detach().cpu())
    return torch.cat(chunks, dim=0)



def collect_regression_targets(dataset, target: str = "criteria_labels") -> torch.Tensor:
    if target == "criteria_labels":
        if not hasattr(dataset, "criteria_labels_all"):
            raise ValueError("Dataset must expose `criteria_labels_all` for fine-tuning.")
        return dataset.criteria_labels_all.float()
    elif target == "accident_label":
        if not hasattr(dataset, "accident_label_all"):
            raise ValueError("Dataset must expose `accident_label_all` for fine-tuning.")
        return dataset.accident_label_all.float().unsqueeze(1)
    else:
        raise ValueError(f"Unsupported target variable: {target}")
    

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


def masked_rmse_loss(pred, target):
    mask = target != -1
    if mask.sum() == 0:
        return pred.sum() * 0.0
    diff = pred[mask] - target[mask]
    return torch.sqrt(torch.mean(diff ** 2))


def masked_mae_loss(pred, target):
    mask = target != -1
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return torch.mean(torch.abs(pred[mask] - target[mask]))


def masked_mse_loss(pred, target):
    mask = target != -1
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return torch.mean((pred[mask] - target[mask]) ** 2)


def masked_bce_with_logits_loss(logits, target):
    mask = target != -1
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        logits[mask],
        target[mask].float(),
        reduction="mean",
    )

