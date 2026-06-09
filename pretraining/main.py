import torch
from tqdm import tqdm
import os

from pretraining.utils import set_seed, evaluate_embeddings, _is_valid_number, _safe_mean
from pretraining.config import Configuration, setup_config
from pretraining.models import InterRoute2Vec
from pretraining.metrics import compute_eval_score, get_metrics
from dataloader.loader_utils import get_model_data_split


def evaluate(model, dataloader, model_name, debug=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for iter, batch in enumerate(tqdm(dataloader, desc=f"{model_name} Evaluating")):
            out = model(batch)
            losses = out["normalized_losses"]
            loss = sum(losses)
            total_loss += loss.item()
        
            if debug and iter >= 2:
                break
    
    return total_loss / (iter+1), [l.item() for l in losses]

def train_epoch(model, dataloader, optimizer, epoch, debug=False):
    model.train()
    for i, batch in enumerate(pbar := tqdm(dataloader, desc=f"Epoch {epoch}")):
        out = model(batch)
        losses = out["normalized_losses"]
        mapping_loss_pos = out["mapping_loss_pos"]

        # -----------------------------
        # Combine losses
        # -----------------------------

        loss = sum(losses)

        # -----------------------------
        # Backprop
        # -----------------------------

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

 
        pbar.set_postfix(train_loss=f"{loss.item():.4f}")

        if debug and i >= 2:
            break

    return {"loss": loss.item(), "losses": [float(f'{l.item():.4f}') for l in losses], "mapping_loss_pos": mapping_loss_pos}


def main(args, train_loader, val_loader, test_loader):
    args.losses = eval(args.losses) if isinstance(args.losses, str) else args.losses
    set_seed(args.seed)


    sample = next(iter(train_loader))

    args.device = device = args.device if torch.cuda.is_available() else 'cpu'
    
    # from sample, get node_dim, cell_dim, edge_dim for initializing the model
    args.node_dim = sample['node_seq_emb'].shape[-1]
    args.cell_dim = sample['cell_seq_emb'].shape[-1]
    args.edge_dim = sample['edge_seq_emb'].shape[-1]

    # transport modes
    args.num_modes_nodes = sample['node_feasible_modes'].shape[-1]
    args.num_modes_edges = sample['edge_modes'].shape[-1]

    debug = vars(args).get("debug", False)

    # Initialize model
    model = InterRoute2Vec(args).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_parameters = num_params_model
    print(f"Model initialized with {num_parameters:,} trainable parameters.")
    
    
    checkpoint_dir = "Insert: checkpoint directory for the pretrained model"

    os.makedirs(checkpoint_dir, exist_ok=True)
    epochs = args.max_epochs
    best_model_path = "Insert: path to save the best model"

    
    # Training
    patience = args.patience
    best_val = {"loss": float('inf'), "losses": None, "weighted_parts": None, "task_weights": None,
                "train_losses": [], "val_losses": [], "val_scores": [], "val_best_losses": {}, "val_best_scores": {}, "val_metrics": None, "eval_scores": None, 
                "test_metrics": None, "test_scores": None}
    
    try:
        print("Starting training ...")
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(model=model, dataloader=train_loader, optimizer=optimizer, epoch=epoch, debug=debug)
            best_val["mapping_loss_pos"] = train_metrics["mapping_loss_pos"]

            # validation
            val_loss, losses = evaluate(model=model, dataloader=val_loader, model_name=args.model_name, debug=debug)

            print(f'[{args.model_name}] Epoch {epoch} - Validation Loss: {val_loss:.4f} - Train Loss: {train_metrics["loss"]:.4f}.')

            best_val["train_losses"].append(train_metrics["loss"])
            
            best_val["val_losses"].append(val_loss)

            # Early stopping based on metrics
            val_metrics = get_metrics(model=model, dataloader=val_loader, debug=debug)
            val_scores = compute_eval_score(args, val_metrics)
            val_score = val_scores['score']
            best_val["val_scores"].append(val_score)

            if val_score > best_val.get('val_score', float('-inf')) or best_val.get('val_score', float('-inf')) is None:
                print(f'New best model found at epoch {epoch}. Validation score increased from {best_val.get("val_score", float("-inf")):.4f} to {val_score:.4f}. Saving model...')
                best_val["val_score"] = val_score
                best_val["loss"] = val_loss
                best_val["losses"] = losses

                best_val["val_best_losses"][epoch] = float(f'{val_loss:.4f}')
                best_val["val_best_scores"][epoch] = float(f'{val_score:.4f}')

                patience = args.patience  # reset patience counter
                # Save the best model
                torch.save(model.state_dict(), best_model_path)

                if test_loader is not None:
                    print("Getting test metrics for best model ...")
                    best_test_metrics = get_metrics(model=model, dataloader=test_loader, debug=debug)
                    test_scores = compute_eval_score(args, best_test_metrics)
                    best_val["test_metrics"] = best_test_metrics
                    best_val["test_scores"] = test_scores
                best_val["val_metrics"] = val_metrics
                best_val["eval_scores"] = val_scores
            else:
                patience -= 1
                print(f'best validation score: {best_val["val_score"]:.4f}. Patience remaining: {patience}')
                if patience <= 0:
                    print(f'Early stopping triggered after {epoch} epochs.')
                    break

        print(f"Best validation score: {best_val['val_score']:.4f}. Best model saved at {best_model_path}.")
    except KeyboardInterrupt as e:
        print("Training interrupted by user.")


if __name__ == '__main__':
    exp_config = Configuration() 
    setup_config(exp_config)

    expnum = 1
    tot_exp = len(exp_config.get_configs())
    print('Number of experiments: ', tot_exp)
    for args in exp_config.get_configs():   
        print(f'Starting experiment number {expnum}/{tot_exp} ...')  
        args.expnum = expnum  
        expnum += 1    

        train_loader, val_loader, test_loader = get_model_data_split(args)
        main(args=args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)