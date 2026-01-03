import os, gc, json, random, warnings, argparse, joblib
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb

from preprocess import preprocess_data, build_preprocessor, NUM_COLS_90, BIN_COLS_90, filter_treatment_episodes 
from train.dataset import WarfarinDataset
from Warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
from train.train import train_model

TARGET = "inr"
TIME_COLS = [ "treatment_day",]
DT_INDEX = 0  # We use the first channel of our time vector as the absolute timestamp

warnings.filterwarnings("ignore", category=RuntimeWarning)
# These settings help the GPU run the LSTM math as fast as possible
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN = (DEVICE.type == "cuda")
# Standard parameters for efficient data loading
LOADER_KW = dict(num_workers=2, pin_memory=PIN, persistent_workers=False, prefetch_factor=2)

OUTPUT_DIR ="real_NET_outputs_nested_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", os.path.abspath(OUTPUT_DIR))
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def pct_within_20pct(y_true, y_pred):
    """Calculates what percentage of predictions are within 20% of the true INR."""
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom) <= 0.20))

def _wandb_init_compat(**kw):
    """Ensures a clean Weights & Biases connection for logging."""
    try: return wandb.init(finish_previous=True, **kw)
    except TypeError: return wandb.init(**kw)


def nested_cv_pipeline(
    df_path: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 20,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    bin_quantiles: int = 10,
    val_frac_outer_train: float = 0.1,
    specific_fold: int = None,
):
    """
    This is the main orchestration script. It uses nested cross-validation:
    the outer loop measures final performance, while the inner loop 
    optimizes model settings (hyperparameters).
    """
    print(f"CUDA available: {torch.cuda.is_available()}  |  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        dev_id = torch.cuda.current_device()
        print(f"CUDA {dev_id}: {torch.cuda.get_device_name(dev_id)} CC{torch.cuda.get_device_capability(dev_id)}")

    df_raw = pd.read_csv(df_path)
    # We preserve demographics here to ensure the final CSVs can be analyzed by group later
    df = preprocess_data(df_raw, preserve_demographics=True)
    df = filter_treatment_episodes(
        df, 
        patient_id_col='subject_id', 
        date_col='inr_datetime',    
        max_gap_days=100,
        inr_col='inr'
    )

    drop_cols = ['subject_id', TARGET, ] + TIME_COLS 
    X_df_full = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X_time_full = df[TIME_COLS].values.astype(np.float32)
    y_full = df[TARGET].values.astype(np.float32)
    groups_full = df['subject_id'].values
    # Stratifying by INR bins helps ensure each fold sees a full range of patient results
    y_bins_full = pd.qcut(y_full, q=bin_quantiles, labels=False, duplicates='drop')

    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    fold_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_df_full, y_bins_full, groups_full), start=1):
        if specific_fold is not None and outer_fold != specific_fold:
            continue

        print(f"\nOuter Fold {outer_fold}/{outer_splits}")

        # We fit the preprocessor ONLY on the training data to avoid 'future leakage'
        preprocessor_outer = build_preprocessor(NUM_COLS_90, BIN_COLS_90)
        X_train = preprocessor_outer.fit_transform(X_df_full.iloc[train_idx])
        X_test  = preprocessor_outer.transform(X_df_full.iloc[test_idx])

        prep_path = os.path.join(OUTPUT_DIR, f"net_preprocessor_outer_fold_{outer_fold}.pkl")
        joblib.dump(preprocessor_outer, prep_path)

        y_train = y_full[train_idx]; y_test = y_full[test_idx]
        t_train = X_time_full[train_idx]; t_test = X_time_full[test_idx]
        g_train = groups_full[train_idx]; g_test = groups_full[test_idx]

        y_bins_train = pd.qcut(y_train, q=bin_quantiles, labels=False, duplicates='drop')
        if len(np.unique(y_bins_train)) < inner_splits:
            print(f"[WARN] Not enough bins for inner CV in fold {outer_fold}. Skipping.")
            continue

        # We pre-split and pre-process the inner folds to make the hyperparameter search much faster
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        X_df_outer_train = X_df_full.iloc[train_idx]
        precomputed = []
        for inner_k, (tr_pos, va_pos) in enumerate(inner_cv.split(np.zeros_like(y_train), y_bins_train, groups=g_train)):
            preprocessor_inner = build_preprocessor(NUM_COLS_90, BIN_COLS_90)
            X_tr = preprocessor_inner.fit_transform(X_df_outer_train.iloc[tr_pos])
            X_va = preprocessor_inner.transform(X_df_outer_train.iloc[va_pos])

            precomputed.append(dict(
                X_tr=X_tr, X_val=X_va,
                y_tr=y_train[tr_pos], y_val=y_train[va_pos],
                t_tr=t_train[tr_pos], t_val=t_train[va_pos],
                g_tr=g_train[tr_pos], g_val=g_train[va_pos],
            ))
            del preprocessor_inner, X_tr, X_va
            gc.collect()

        # MedianPruner kills trials that look like they are performing poorly early on
        pruner = MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(direction="minimize", pruner=pruner, sampler=TPESampler(seed=SEED))

        def objective(trial: optuna.trial.Trial):
            """This function is what Optuna calls to test a specific set of model settings."""
            window_size   = trial.suggest_int("window_size", 3, 20)
            hidden_size   = trial.suggest_int("hidden_size", 16, 256, log=True) 
            num_layers    = trial.suggest_int("num_layers", 2, 5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
            dropout       = trial.suggest_float("dropout", 0.1, 0.5)
            
            scores = []
            for k, pack in enumerate(precomputed):
                # We turn the precomputed arrays into actual PyTorch Dataset objects here
                train_ds = WarfarinDataset(pack["X_tr"], pack["y_tr"], pack["t_tr"], 
                          group_ids=pack["g_tr"], window_size=window_size, 
                          allow_padding=True)
                val_ds = WarfarinDataset(pack["X_val"], pack["y_val"], pack["t_val"], 
                                        group_ids=pack["g_val"], window_size=window_size,
                                        allow_padding=True) 
                
                if len(train_ds) == 0 or len(val_ds) == 0:
                    trial.report(float("inf"), step=k)
                    if trial.should_prune(): raise optuna.TrialPruned()
                    continue

                train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **LOADER_KW)
                val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **LOADER_KW)

                model = NETLSTM( 
                    input_dim=pack["X_tr"].shape[-1],
                    time_dim=pack["t_tr"].shape[-1],
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_pack=True,
                    dt_index=DT_INDEX,
                ).to(DEVICE)

                opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
                crit = nn.MSELoss()

                # Train the model and get the best score it achieved on the validation split
                best_rmse, _, _ = train_model(
                    model=model,
                    train_loader=train_ld,
                    val_loader=val_ld,
                    optimizer=opt,
                    criterion=crit,
                    device=DEVICE,
                    trial_number=f"fold{outer_fold}_trial{trial.number}_split{k}",
                    max_epochs=max_epochs,
                    patience=patience,
                    wandb_run=None,
                    output_dir=OUTPUT_DIR,
                    use_amp=True,
                    save_checkpoints=False,
                )
                scores.append(best_rmse)
                del model, opt, train_ld, val_ld, train_ds, val_ds
                torch.cuda.empty_cache(); gc.collect()

                trial.report(best_rmse, step=k)
                if trial.should_prune(): raise optuna.TrialPruned()

            if not scores: raise optuna.TrialPruned()
            return float(np.mean(scores))

    # Optuna runs the trials defined above to find the optimal params
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        best = study.best_params
        print(f"[Fold {outer_fold}] Best: {json.dumps(best, indent=2)}")

        # Now we retrain the final model using those best params on all training data
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_outer_train, random_state=SEED)
        tr_sub, va_sub = next(gss.split(X_train, y_train, groups=g_train))
        train_ds_final = WarfarinDataset(X_train[tr_sub], y_train[tr_sub], t_train[tr_sub], 
                                group_ids=g_train[tr_sub], window_size=best["window_size"],
                                allow_padding=True)  
        val_ds_final = WarfarinDataset(X_train[va_sub], y_train[va_sub], t_train[va_sub], 
                                    group_ids=g_train[va_sub], window_size=best["window_size"],
                                    allow_padding=True)
        test_ds = WarfarinDataset(X_test, y_test, t_test, group_ids=g_test, 
                                window_size=best["window_size"],
                                allow_padding=True)

        train_ld = DataLoader(train_ds_final, batch_size=batch_size, shuffle=True,  **LOADER_KW)
        val_ld   = DataLoader(val_ds_final,   batch_size=batch_size, shuffle=False, **LOADER_KW)
        test_ld  = DataLoader(test_ds,        batch_size=batch_size, shuffle=False, **LOADER_KW)

        model = NETLSTM(
            input_dim=X_train.shape[-1],
            time_dim=t_train.shape[-1],
            hidden_size=best["hidden_size"],
            num_layers=best["num_layers"],
            dropout=best["dropout"],
            use_pack=True,
            dt_index=DT_INDEX,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=best["learning_rate"])
        criterion = nn.MSELoss()

        run = _wandb_init_compat(project="warfarinNET-nested-cv", name=f"test_NET_outer_fold_{outer_fold}",
                                 config={"outer_fold": outer_fold, **best}, reinit=True)

        # Final train call for the best model on this fold
        train_model(
            model=model,
            train_loader=train_ld,
            val_loader=val_ld,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            trial_number=f"outer_fold_{outer_fold}",
            max_epochs=max_epochs,
            patience=patience,
            wandb_run=run,
            output_dir=OUTPUT_DIR,
            use_amp=True,
            save_checkpoints=True,
            model_name="NET",
            save_attention_every_n_epochs=20,
        )

        # Evaluation phase on the held-out test set
        model.eval()
        preds, targets, indices = [], [], []
        with torch.no_grad():
            for batch in test_ld:
                x, t, yb = batch["features"].to(DEVICE), batch["time"].to(DEVICE), batch["label"].to(DEVICE)
                out = model(x, t)
                if isinstance(out, (tuple, list)): out = out[0]
                preds.extend(out.detach().cpu().numpy())
                targets.extend(yb.detach().cpu().numpy())
                # We save original indices so we can link these predictions back to specific patients
                if "original_indices" in batch:
                    indices.extend(batch["original_indices"].view(-1).cpu().numpy())

        preds_arr = np.asarray(preds, dtype=np.float32)
        targets_arr = np.asarray(targets, dtype=np.float32)

        rmse_score = rmse(targets_arr, preds_arr)
        mae_score  = float(mean_absolute_error(targets_arr, preds_arr))
        w20_score  = pct_within_20pct(targets_arr, preds_arr)
        print(f"[Fold {outer_fold}] Test RMSE={rmse_score:.4f} | MAE={mae_score:.4f} | W20={w20_score:.3%}")
        run.log({"outer_test_rmse": rmse_score, "outer_test_mae": mae_score, "outer_test_within20": w20_score})

        from preprocess import create_demographic_metadata

        pred_df = pd.DataFrame({
            "fold": outer_fold, 
            "y_true": targets_arr.astype(float), 
            "y_pred": preds_arr.astype(float)
        })

        # We attach patient metadata here so we can check for bias in our results later
        if len(indices) > 0:
            demo_metadata = create_demographic_metadata(df.iloc[test_idx], indices)
            for key, values in demo_metadata.items():
                if len(values) == len(pred_df):
                    pred_df[key] = values

        pred_path = os.path.join(OUTPUT_DIR, f"NET_predictions_outer_fold_{outer_fold}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"[INFO] Saved predictions: {pred_path}")

        # Saving the model and its configuration so we can reload it for future predictions
        ckpt_path = os.path.join(OUTPUT_DIR, f"NET_best_model_outer_fold_{outer_fold}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "preprocessor_path": prep_path,
                "input_dim": X_train.shape[-1],
                "time_dim": t_train.shape[-1],
                **best
            }
        }, ckpt_path)
        print(f"[INFO] Saved model: {ckpt_path}")

        run.finish()
        del model, optimizer, criterion, train_ld, val_ld, test_ld
        torch.cuda.empty_cache(); gc.collect()

        fold_results.append({"fold": outer_fold, "rmse": rmse_score, "mae": mae_score, "within20": w20_score})
        fold_result = {"fold": outer_fold, "rmse": rmse_score, "mae": mae_score, "within20": w20_score}
        individual_result_path = os.path.join(OUTPUT_DIR, f"fold_{outer_fold}_result.csv")
        pd.DataFrame([fold_result]).to_csv(individual_result_path, index=False)

        # If we are running the full loop, combine all fold results into a single summary file
        if specific_fold is None: 
            all_results = []
            for f in range(1, outer_splits + 1):
                result_file = os.path.join(OUTPUT_DIR, f"fold_{f}_result.csv")
                if os.path.exists(result_file):
                    fold_df = pd.read_csv(result_file)
                    all_results.append(fold_df.iloc[0].to_dict())
            
            if all_results:
                final_df = pd.DataFrame(all_results)
                final_df.to_csv(os.path.join(OUTPUT_DIR, "NET_nested_cv_summary_metrics.csv"), index=False)

    res = pd.DataFrame(fold_results)
    if not res.empty:
        res.to_csv(os.path.join(OUTPUT_DIR, "NET_nested_cv_summary_metrics.csv"), index=False)
        print("\nNested CV Summary")
        print(res)
        print(f"\nAverage RMSE: {res['rmse'].mean():.4f} ± {res['rmse'].std():.4f}")
    else:
        print("[INFO] No folds were run.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", type=str, default="warfarin.csv")
    ap.add_argument("--outer_splits", type=int, default=5)
    ap.add_argument("--inner_splits", type=int, default=3)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()

    nested_cv_pipeline(
        df_path=args.df,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        n_trials=args.trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        specific_fold=args.fold,
    )