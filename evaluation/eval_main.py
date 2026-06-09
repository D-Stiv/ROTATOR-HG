import argparse
import os
import pickle
from tqdm import tqdm
import copy
import torch

from pretraining.utils import set_seed

from evaluation.eval_config import Configuration, setup_config
from evaluation.eval_utils import build_model, model_forward_for_embedding
from evaluation.eval_metrics import compute_unsupervised_batch_metrics, compute_semi_supervised_batch_metrics, summarize_metric_dicts
from evaluation.finetune import run_finetune_and_supervised



def evaluate_unsup_and_semi(model, test_loader, model_name: str, only_struct: bool, spatio_struct: bool):
    test_dataset = test_loader.dataset
    unsup_metrics = []
    semi_metrics = []

    for batch in tqdm(test_loader, desc=f"[{model_name}] Evaluating"):
        embeddings = model_forward_for_embedding(model, batch)
        unsup_metrics.append(compute_unsupervised_batch_metrics(batch, embeddings, model_name, only_struct=only_struct, spatio_struct=spatio_struct))
        semi_metrics.append(compute_semi_supervised_batch_metrics(batch, embeddings, model, model_name, test_dataset))

    results = {
        "model_name": model_name,
        "unsupervised": summarize_metric_dicts(unsup_metrics),
        "semi_supervised": summarize_metric_dicts(semi_metrics),
    }
    return results
    
def evaluate_model_type(args, eval_type, model_loaders):
    device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader = model_loaders

    model = build_model(args)

    results = {}

    if eval_type in ["un_semi"]:
        un_semi_results = evaluate_unsup_and_semi(model, test_loader)

        results["unsupervised"] = un_semi_results.get("unsupervised", {})
        results["semi_supervised"] = un_semi_results.get("semi_supervised", {})

        
    elif eval_type == "finetune-supervised":
        debug_str = "_debug" if getattr(args, "debug", False) else ""
        finetune_ckpt_dir = "Insert: Directory to save finetuning checkpoints" + debug_str
        os.makedirs(finetune_ckpt_dir, exist_ok=True)


        val_metrics, test_metrics = run_finetune_and_supervised(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            finetune_ckpt_dir=finetune_ckpt_dir,
            device=device,
        )

        results["finetune_test"] = test_metrics
        results["finetune_val"] = val_metrics
    
    else:
        raise ValueError(f"Unsupported evaluation type: {eval_type}.")

    return results


def parse_comma_separated(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main(args, model_loaders):


    evaluate_model_type(
            args,
            eval_type=args.eval_type,
            model_loaders=model_loaders,
        )

    print("\nFinished evaluations ...")

if __name__ == "__main__":
    exp_config = Configuration() 
    setup_config(exp_config)

    exp_num = 1
    tot_exp = len(exp_config.get_configs())
    print('Number of experiments: ', tot_exp)


    for args in exp_config.get_configs():   
        print(f'Starting experiment number {exp_num}/{tot_exp} ...')  
        args.expnum = exp_num  
        exp_num += 1    

        set_seed(args.seed)

        args.device = args.device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {args.device}")


        model_name = args.model_name.lower()
        from dataloader.loader_utils import get_model_data_split


        train_loader, val_loader, test_loader = get_model_data_split(args)

        main(args=args, model_loaders=(train_loader, val_loader, test_loader))
