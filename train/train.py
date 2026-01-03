import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from contextlib import nullcontext
from typing import Optional
import gc

def _make_autocast(use_amp: bool, device: torch.device):
    """
    Sets up 'Autocast' for mixed precision training.
    This allows the model to use lower-precision numbers where possible, 
    saving memory and speeding up the GPU without losing accuracy.
    """
    if not use_amp or device.type != "cuda":
        return nullcontext()
    try:
        from torch import amp
        return amp.autocast("cuda")
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast()

def _make_grad_scaler(use_amp: bool, device: torch.device):
    """
    The companion to Autocast. It scales gradients to prevent 
    'underflow' (where numbers become too small for the computer to track).
    """
    if not use_amp or device.type != "cuda":
        return None
    try:
        from torch import amp
        return amp.GradScaler("cuda")
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=True)

def rmse(y_true, y_pred):
    """Calculates the standard Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def pct_within_20pct(y_true, y_pred):
    """
    A clinical metric: what percentage of our dose predictions 
    are within 20% of the actual dose?
    """
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom) <= 0.20))

def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    trial_number: Optional[str] = "0",
    max_epochs: int = 100,
    patience: int = 10,
    wandb_run=None,
    output_dir: str = "outputs",
    model_name: str = "WarfarinNet",
    use_amp: bool = True,
    save_checkpoints: bool = True,
    cuda_debug_check: bool = True,
    save_attention_every_n_epochs: int = 5,
    metadata_dict: dict = None, 
):
    """
    Handles the heavy lifting of training the model.
    It tracks the best performance and saves attention maps so we can 
    interpret the model's decisions later.
    """
    # Create a specific folder for this model's results
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    scaler = _make_grad_scaler(use_amp, device)
    best_val_rmse = float("inf")
    last_val_mae = float("inf")
    last_val_pct20 = 0.0
    counter = 0 # Tracks how many epochs we've gone without improvement
    best_epoch = -1
    best_state = None
    start_time = time.time()
    printed_cuda_info = False
    
    best_attention_data = None

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        batch_train_losses = []

        # TRAINING PHASE
        for batch in train_loader:
            x = batch['features'].to(device, non_blocking=True)
            t = batch['time'].to(device, non_blocking=True)
            y = batch['label'].to(device, non_blocking=True)

            if cuda_debug_check and device.type == "cuda" and not printed_cuda_info:
                print(f"✔ Using device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
                printed_cuda_info = True

            optimizer.zero_grad(set_to_none=True)
            
            # Wrap forward pass in autocast for speed
            with _make_autocast(use_amp, device):
                if getattr(model, "attention", False):
                    # Request attention weights along with the prediction
                    output, _, _ = model(x, t, return_attention=True)
                else:
                    output = model(x, t)
                
                output = output.view(-1, 1)
                y = y.view(-1, 1)
                loss = criterion(output, y)

            # Backpropagation step
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            batch_train_losses.append(loss.item())

        train_loss = float(np.mean(batch_train_losses)) if batch_train_losses else float("nan")

        # VALIDATION PHASE
        model.eval()
        preds, targets, val_batch_losses = [], [], []
        all_time_attn, all_feat_attn = [], []
        all_batch_indices = []
        
        with torch.no_grad(), _make_autocast(use_amp, device):
            for batch_idx, batch in enumerate(val_loader):
                x = batch['features'].to(device, non_blocking=True)
                t = batch['time'].to(device, non_blocking=True)
                y = batch['label'].to(device, non_blocking=True)
                
                if getattr(model, "attention", False):
                    out, time_attn, feat_attn = model(x, t, return_attention=True)
                    
                    # We move attention weights to CPU to save GPU memory
                    if time_attn is not None:
                        time_attn_cpu = time_attn.detach().cpu().numpy()
                        # If we have multi-head attention, we average it for easier viewing
                        if time_attn_cpu.ndim == 4: 
                            time_attn_cpu = time_attn_cpu.mean(axis=1) 
                        all_time_attn.append(time_attn_cpu)
                        
                    if feat_attn is not None:
                        all_feat_attn.append(feat_attn.detach().cpu().numpy())
                        
                    all_batch_indices.extend([batch_idx] * len(out))
                else:
                    out = model(x, t)
                    
                out = out.view(-1, 1)
                y = y.view(-1, 1)
                vloss = criterion(out, y)

                val_batch_losses.append(vloss.item())
                preds.extend(np.atleast_1d(out.cpu().numpy()))
                targets.extend(np.atleast_1d(y.cpu().numpy()))

        # Calculate final epoch metrics
        val_loss = float(np.mean(val_batch_losses))
        preds_arr = np.asarray(preds, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.float32)
        rmse_score = rmse(targets_arr, preds_arr)
        last_val_mae = float(mean_absolute_error(targets_arr, preds_arr))
        last_val_pct20 = pct_within_20pct(targets_arr, preds_arr)

        # Decide if we should save the attention weights for this epoch
        save_attention_now = (
            (getattr(model, "attention", False)) and 
            (rmse_score < best_val_rmse or epoch % save_attention_every_n_epochs == 0)
        )
        
        if save_attention_now:
            attention_data = {
                'preds': preds_arr,
                'targets': targets_arr,
                'epoch': epoch,
                'val_rmse': rmse_score,
                'val_mae': last_val_mae,
                'batch_indices': np.array(all_batch_indices),
            }
            
            # Combine all small batch attention arrays into one big array
            if all_time_attn:
                attention_data['attn_time'] = np.concatenate(all_time_attn, axis=0)
            if all_feat_attn:
                attention_data['attn_feat'] = np.concatenate(all_feat_attn, axis=0)
            
            # Merge in demographic info (like age/sex) if provided
            if metadata_dict:
                attention_data.update(metadata_dict)
            
            # If this is the best version of the model so far, save it as 'best'
            if rmse_score < best_val_rmse:
                np.savez_compressed(
                    os.path.join(output_dir, f"attn_outputs_trial_{trial_number}_best.npz"),
                    **attention_data
                )
                best_attention_data = attention_data.copy()
                
            # Periodic snapshots help us see how attention evolves over time
            if epoch % save_attention_every_n_epochs == 0:
                np.savez_compressed(
                    os.path.join(output_dir, f"attn_outputs_trial_{trial_number}_epoch_{epoch}.npz"),
                    **attention_data
                )

        # Log stats to Weights & Biases if connected
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": rmse_score,
                "val_mae": last_val_mae,
                "val_pct_within_20pct": last_val_pct20
            })

        # Check for improvement; stop early if the model stops getting better
        if rmse_score < best_val_rmse:
            best_val_rmse = rmse_score
            best_epoch = epoch
            counter = 0
            if save_checkpoints:
                # We move weights to CPU before saving to keep the checkpoint portable
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(best_state, f"{output_dir}/best_model_trial_{trial_number}.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{max_epochs} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val RMSE: {rmse_score:.4f} "
            f"MAE: {last_val_mae:.4f} "
            f"W20: {last_val_pct20*100:.2f}% "
            f"Epoch time: {epoch_time:.2f}s"
        )
        
        # Explicitly clean up memory to prevent GPU crashes
        if 'all_time_attn' in locals(): del all_time_attn
        if 'all_feat_attn' in locals(): del all_feat_attn
        torch.cuda.empty_cache()
        gc.collect()

    # Reload the best weights and save a final summary
    if save_checkpoints and best_state is not None:
        model.load_state_dict(best_state, strict=False)
        torch.save(model.state_dict(), f"{output_dir}/final_model_trial_{trial_number}.pt")

    if best_attention_data and getattr(model, "attention", False):
        summary_path = os.path.join(output_dir, f"attention_summary_trial_{trial_number}.npz")
        np.savez_compressed(summary_path, **best_attention_data)
        print(f"[INFO] Saved attention summary to: {summary_path}")

    print(
        f"Training completed in {(time.time()-start_time)/60:.2f} minutes. "
        f"Best val RMSE: {best_val_rmse:.4f} at epoch {best_epoch+1}"
    )
    
    return best_val_rmse, last_val_mae, last_val_pct20