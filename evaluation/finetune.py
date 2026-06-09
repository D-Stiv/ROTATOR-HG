import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
from tqdm import tqdm
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


from evaluation.eval_utils import _safe_mean_std, collect_embeddings, collect_regression_targets
from evaluation.eval_metrics import _multilabel_batch_metrics, _regression_batch_metrics
from evaluation.finetune_models import TwoLayerMLP


def get_metrics(model, emb, y, batch_size, device, target, task_names=None, threshold=0.5, y_acc=None, proxy_pos=None):
    if y.ndim == 1:
        y = y.unsqueeze(-1)

    ds_tensors = [emb, y]

    if y_acc is not None:
        if not isinstance(y_acc, torch.Tensor):
            y_acc = torch.as_tensor(y_acc)

        if y_acc.ndim == 1:
            y_acc = y_acc.unsqueeze(-1)

        num_tasks = y.shape[1]
        if proxy_pos is None or not (0 <= proxy_pos <= num_tasks - 1):
            raise ValueError(
                f"proxy_pos must be in [0, {num_tasks - 1}] when y_acc is provided, got {proxy_pos}"
            )

        ds_tensors.append(y_acc)

    ds = TensorDataset(*ds_tensors)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if task_names is None:
        task_names = [f"task_{i}" for i in range(y.shape[1])]

    count_metric_names = {"tp", "fp", "tn", "fn"}

    def aggregate_metric_history(batch_metrics, per_task_history):
        overall = {}
        if len(batch_metrics) > 0:
            for metric_name in batch_metrics[0].keys():
                vals = [
                    m[metric_name]
                    for m in batch_metrics
                    if m[metric_name] is not None and not np.isnan(m[metric_name])
                ]

                if metric_name in count_metric_names:
                    overall[metric_name] = int(np.sum(vals)) if len(vals) > 0 else 0
                else:
                    mean, std = _safe_mean_std(vals)
                    overall[f"{metric_name}_mean"] = mean
                    overall[f"{metric_name}_std"] = std

        per_task = {}
        for task_name, metric_dict in per_task_history.items():
            per_task[task_name] = {}
            for metric_name, vals in metric_dict.items():
                clean_vals = [v for v in vals if v is not None and not np.isnan(v)]

                if metric_name in count_metric_names:
                    per_task[task_name][metric_name] = int(np.sum(clean_vals)) if len(clean_vals) > 0 else 0
                else:
                    mean, std = _safe_mean_std(clean_vals)
                    per_task[task_name][f"{metric_name}_mean"] = mean
                    per_task[task_name][f"{metric_name}_std"] = std

        return overall, per_task

    batch_metrics = []
    per_task_history = {}
    loss = []

    proxy_batch_metrics = []
    proxy_per_task_history = {}
    proxy_loss = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if y_acc is None:
                xb, yb = batch
                yb_acc = None
            else:
                xb, yb, yb_acc = batch

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss.append(model.get_loss(pred, yb).item())

            if "label" in target:
                batch_result = _multilabel_batch_metrics(
                    pred,
                    yb,
                    threshold=threshold,
                    task_names=task_names,
                )
            else:
                pred = torch.sigmoid(pred)
                batch_result = _regression_batch_metrics(
                    pred,
                    yb,
                    task_names=task_names,
                )

            batch_metrics.append(batch_result["macro"])

            for task_name, metrics in batch_result["per_task"].items():
                if task_name not in per_task_history:
                    per_task_history[task_name] = {k: [] for k in metrics.keys()}
                for metric_name, metric_value in metrics.items():
                    per_task_history[task_name][metric_name].append(metric_value)

            if yb_acc is not None:
                yb_acc = yb_acc.to(device)

                pred_acc = pred[:, proxy_pos]
                if pred_acc.ndim == 1:
                    pred_acc = pred_acc.unsqueeze(-1)

                if yb_acc.ndim == 1:
                    yb_acc = yb_acc.unsqueeze(-1)

                proxy_task_name = ["accident_proxy"]

                if "label" in target:
                    proxy_loss.append(model.get_loss(pred_acc, yb_acc).item())
                    proxy_batch_result = _multilabel_batch_metrics(
                        pred_acc,
                        yb_acc,
                        threshold=threshold,
                        task_names=proxy_task_name,
                    )
                else:
                    proxy_loss.append(model.get_loss(pred_acc, yb_acc).item())
                    proxy_batch_result = _regression_batch_metrics(
                        pred_acc,
                        yb_acc,
                        task_names=proxy_task_name,
                    )

                proxy_batch_metrics.append(proxy_batch_result["macro"])

                for task_name, metrics in proxy_batch_result["per_task"].items():
                    if task_name not in proxy_per_task_history:
                        proxy_per_task_history[task_name] = {k: [] for k in metrics.keys()}
                    for metric_name, metric_value in metrics.items():
                        proxy_per_task_history[task_name][metric_name].append(metric_value)

    overall, per_task = aggregate_metric_history(batch_metrics, per_task_history)
    loss = np.mean(loss) if len(loss) > 0 else 0.0

    out = {
        "overall": overall,
        "per_task": per_task,
        "per_batch": batch_metrics,
        "loss": loss,
    }

    if y_acc is not None:
        proxy_overall, proxy_per_task = aggregate_metric_history(
            proxy_batch_metrics,
            proxy_per_task_history,
        )
        out["accident_proxy"] = {
            "proxy_pos": proxy_pos,
            "overall": proxy_overall,
            "per_task": proxy_per_task,
            "per_batch": proxy_batch_metrics,
            "loss": np.mean(proxy_loss) if len(proxy_loss) > 0 else 0.0,
        }

    return out


def train_finetune_head(model, train_loader, val_loader, test_loader, finetune_ckpt_path, batch_size, target, 
                        hidden_dim, dropout, lr, weight_decay, epochs, patience,
                        device, regression_loss="mse", task_names=None) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:

    train_emb = collect_embeddings(model, train_loader)
    val_emb = collect_embeddings(model, val_loader)
    test_emb = collect_embeddings(model, test_loader)

    # normalize embeddings
    mean = train_emb.mean(0, keepdim=True)
    std = train_emb.std(0, keepdim=True) + 1e-8

    train_emb = (train_emb - mean) / std
    val_emb   = (val_emb - mean) / std
    test_emb  = (test_emb - mean) / std

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    test_dataset = test_loader.dataset

    assert target == "criteria_labels", "For supervised evaluation, target must be 'criteria_labels'."

    all_task_names = ['quiet', 'scenic', 'bike', 'safe', 'eco', 'social', 'walk', 'nature', 'proxy', 'weather']
    # sort all_task_names order
    all_task_names.sort()

    train_y = collect_regression_targets(train_dataset, target=target)
    val_y = collect_regression_targets(val_dataset, target=target)
    test_y = collect_regression_targets(test_dataset, target=target)

    if task_names is None:
        task_names = all_task_names
    task_names = [t for t in all_task_names if t in task_names]
    train_y = train_y[:, [all_task_names.index(t) for t in task_names]]
    val_y = val_y[:, [all_task_names.index(t) for t in task_names]]
    test_y = test_y[:, [all_task_names.index(t) for t in task_names]]
    
    val_y_acc = collect_regression_targets(val_dataset, target="accident_label")
    test_y_acc = collect_regression_targets(test_dataset, target="accident_label")
    proxy_pos = task_names.index('proxy') if 'proxy' in task_names else None

    input_dim = train_emb.shape[1]
    output_dim = train_y.shape[1]

    model_name = model.__class__.__name__

    finetune_args = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "target": target,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "model_name": model_name,
        "task_names": task_names,
    }


    model = TwoLayerMLP(input_dim, hidden_dim, output_dim, dropout=dropout, target=target, regression_loss=regression_loss).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(train_emb, train_y)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    best_val_score = math.inf
    best_val_metric = math.inf
    best_val_loss = math.inf
    patience_count = 0
    best_val_metrics = None
    best_test_metrics = None

    train_losses = []
    val_losses = []
    best_val_losses = []
    best_val_scores = []

    try:
        epoch_pbar = tqdm(range(1, epochs + 1), desc=f"[{model_name}] Training")
        for _epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            for iter, (xb, yb) in enumerate(pbar := tqdm(train_loader, desc=f"[{model_name}] Epoch {_epoch}/{epochs}")):
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = model.get_loss(pred, yb)

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (VERY important)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0
                )
                
                optimizer.step()

                pbar.set_postfix(train_loss=f"{loss.item():.4f}")

                train_loss += loss.item()
            
            # train_loss /= len(train_loader)
            train_loss /= (iter + 1)
            train_losses.append(train_loss)

            model.eval()
            val_metrics = get_metrics(model, val_emb, val_y, batch_size=batch_size, device=device, target=target, task_names=task_names, y_acc=val_y_acc, proxy_pos=proxy_pos)
            val_loss = val_metrics["loss"]
            val_losses.append(val_loss)

            if "label" in target:
                val_metric = val_metrics["overall"]["f1_mean"]
                val_score = -val_metric
                metric = "f1"
            else:
                val_metric = val_metrics["overall"]["rmse_mean"]
                val_score = val_metric
                metric = "rmse"
            
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_metric=f"{val_metric:.4f}"
            )

            # if val_score < best_val_score:
            if val_loss < best_val_loss:
                # print(f"[{model_name}] New best model found at epoch {_epoch} with val {metric} {val_score:.4f} (previous best was {best_val_metric:.4f})")
                print(f"[{model_name}] New best model found at epoch {_epoch} with val loss {val_loss:.4f} (previous best was {best_val_loss:.4f})")
                
                best_val_loss = val_loss
                best_val_score = val_score
                best_val_metric = val_metric
                best_val_metrics = val_metrics
                best_val_losses.append(val_loss)
                best_val_scores.append(val_score)

                torch.save(model.state_dict(), finetune_ckpt_path)

                test_metrics = get_metrics(model, test_emb, test_y, batch_size=batch_size, device=device, target=target, task_names=task_names, y_acc=test_y_acc, proxy_pos=proxy_pos)
                best_test_metrics = test_metrics
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"[{model_name}] No improvement in validation score for {patience_count} epochs. Stopping finetuning.")
                    break
            
            if _epoch == 1:
                print(f"[{model_name}] Epoch {_epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Score: {val_score:.4f}")


    except KeyboardInterrupt:
        print("Finetuning interrupted. Using best model found so far.")
    
    finetune_args["eff_epochs"] = _epoch
    best_val_metrics["val_score_best"] = best_val_score
    best_val_metrics["val_metric_best"] = best_val_metric
    best_val_metrics["val_losses_best"] = best_val_losses
    best_val_metrics["val_scores_best"] = best_val_scores
    best_val_metrics["train_losses"] = train_losses
    best_val_metrics["val_losses"] = val_losses

    return best_val_metrics, best_test_metrics, finetune_args


def run_finetune_and_supervised(model, train_loader, val_loader, test_loader, args, finetune_ckpt_dir: str, device):
    
    task_names = eval(args.supervised_task_names) if getattr(args, "supervised_task_names", None) is not None else None
    finetune_ckpt_path = "Insert: Path to save finetuning checkpoints"

    hidden_dim = args.finetune_hidden_dim
    dropout = args.finetune_dropout
    lr = args.finetune_lr
    weight_decay = args.finetune_weight_decay
    epochs_ = args.finetune_epochs
    patience = args.finetune_patience
    batch_size = args.finetune_batch_size

    finetune_regression_loss = args.finetune_regression_loss


    val_metrics, test_metrics, finetune_args = train_finetune_head(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        finetune_ckpt_path=finetune_ckpt_path,
        batch_size=batch_size,
        target=args.supervised_target,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs_,
        patience=patience,
        device=device,
        regression_loss=finetune_regression_loss,
        task_names=task_names,
    )

    # save finetune args
    with open(os.path.join(finetune_ckpt_dir, f"args.pkl"), "wb") as f:
        pickle.dump(finetune_args, f)

    return val_metrics, test_metrics
    