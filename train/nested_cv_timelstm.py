import os
import gc
import json
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx.utils.backends")

# Settings to optimize training speed on modern GPUs
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from preprocess import preprocess_data, build_preprocessor, NUM_COLS_90, BIN_COLS_90, filter_treatment_episodes
from train.dataset import WarfarinDataset
from time_aware_lstm import TimeLSTM
from train.train import train_model

TARGET = 'inr'  

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs_nested_cv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.environ.setdefault("WANDB_DIR", os.path.abspath(OUTPUT_DIR))
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

PIN = (DEVICE.type == "cuda")
LOADER_KW = dict(num_workers=2, pin_memory=PIN, persistent_workers=False, prefetch_factor=2)

TIME_COLS = ["treatment_day"]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def pct_within_20pct(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-8)
    pct_err = np.abs((y_pred - y_true) / denom)
    return float(np.mean(pct_err <= 0.20))

def _wandb_init_compat(**kwargs):
    """Ensures Weights & Biases connects properly even if a previous run didn't close."""
    try:
        return wandb.init(finish_previous=True, **kwargs)
    except TypeError:
        return wandb.init(**kwargs)

def nested_cv_pipeline(
    df_path: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 50,
    bin_quantiles: int = 10,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 128,
    save_predictions: bool = True,
    val_frac_outer_train: float = 0.1,
    specific_fold: int = None
):
    """
    This is the core pipeline. It uses Nested Cross-Validation to ensure 
    that we aren't 'cheating' by tuning our hyperparameters on the same data 
    we use for final testing.
    """
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {DEVICE}")
    if DEVICE.type == "cuda":
        dev_id = torch.cuda.current_device()
        print(f"CUDA device {dev_id}: {torch.cuda.get_device_name(dev_id)}")

    df_raw = pd.read_csv(df_path)
    df = preprocess_data(df_raw)
    df = filter_treatment_episodes(
        df, 
        patient_id_col='subject_id', 
        date_col='inr_datetime',      
        max_gap_days=100,
        inr_col='inr'
    )

    X_df_full = df.drop(columns=['subject_id', TARGET])
    X_time_full = df[TIME_COLS].values
    y_full = df[TARGET].values
    groups_full = df['subject_id'].values
    y_bins_full = pd.qcut(y_full, q=bin_quantiles, labels=False, duplicates='drop')

    # StratifiedGroupKFold keeps patients together while maintaining balanced INR target bins
    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)

    fold_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_df_full, y_bins_full, groups_full), start=1
    ):
        if specific_fold is not None and outer_fold != specific_fold:
            continue

        print(f"\nOuter Fold {outer_fold}/{outer_splits}")

        # Preprocessor must fit ONLY on train_idx to prevent data leakage from the test set
        preprocessor_outer = build_preprocessor(NUM_COLS_90, BIN_COLS_90)
        X_train = preprocessor_outer.fit_transform(X_df_full.iloc[train_idx])
        X_test  = preprocessor_outer.transform(X_df_full.iloc[test_idx])

        prep_path = os.path.join(OUTPUT_DIR, f"preprocessor_outer_fold_{outer_fold}.pkl")
        joblib.dump(preprocessor_outer, prep_path)

        y_train = y_full[train_idx]
        y_test  = y_full[test_idx]
        t_train = X_time_full[train_idx]
        t_test  = X_time_full[test_idx]
        group_train = groups_full[train_idx]
        group_test  = groups_full[test_idx]

        y_bins_train = pd.qcut(y_train, q=bin_quantiles, labels=False, duplicates='drop')
        
        wandb_run = _wandb_init_compat(
            project="warfarin-nested-cv-no-leak",
            name=f"outer_fold_{outer_fold}",
            config={
                "outer_fold": outer_fold,
                "outer_splits": outer_splits,
                "inner_splits": inner_splits,
                "n_trials": n_trials,
                "batch_size": batch_size,
                "seed": SEED
            },
            reinit=True
        )

        # Precomputing transforms for inner folds to save CPU time during optimization
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        X_df_outer_train = X_df_full.iloc[train_idx]

        precomputed_splits = []
        for split_id, (inner_train_idx, val_idx) in enumerate(
            inner_cv.split(np.zeros_like(y_train), y_bins_train, groups=group_train), start=0
        ):
            preprocessor_inner = build_preprocessor(NUM_COLS_90, BIN_COLS_90)
            X_tr  = preprocessor_inner.fit_transform(X_df_outer_train.iloc[inner_train_idx])
            X_val = preprocessor_inner.transform(X_df_outer_train.iloc[val_idx])

            precomputed_splits.append({
                "X_tr": X_tr, "X_val": X_val,
                "y_tr": y_train[inner_train_idx], "y_val": y_train[val_idx],
                "t_tr": t_train[inner_train_idx], "t_val": t_train[val_idx],
                "g_tr": group_train[inner_train_idx], "g_val": group_train[val_idx]
            })
            gc.collect()

        # Optuna finds the best model settings automatically
        pruner = MedianPruner(n_warmup_steps=15)
        study = optuna.create_study(direction="minimize", pruner=pruner, sampler=TPESampler(seed=SEED))

        def objective(trial: optuna.trial.Trial):
            window_size   = trial.suggest_int("window_size", 3, 20)
            hidden_size   = trial.suggest_int("hidden_size", 16, 128, log=True)
            num_layers    = trial.suggest_int("num_layers", 2, 5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            dropout       = trial.suggest_float("dropout", 0.1, 0.5)
            attention     = trial.suggest_categorical("attention", [False, True])

            inner_scores = []
            for split_idx, split_pack in enumerate(precomputed_splits):
                train_ds = WarfarinDataset(split_pack["X_tr"], split_pack["y_tr"], split_pack["t_tr"], 
                          group_ids=split_pack["g_tr"], window_size=window_size, allow_padding=True) 
                val_ds = WarfarinDataset(split_pack["X_val"], split_pack["y_val"], split_pack["t_val"], 
                                        group_ids=split_pack["g_val"], window_size=window_size, allow_padding=True) 

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **LOADER_KW)
                val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **LOADER_KW)

                model = TimeLSTM(
                    input_dim=split_pack["X_tr"].shape[-1],
                    time_dim=split_pack["t_tr"].shape[-1],
                    hidden_size=hidden_size, num_layers=num_layers,
                    dropout=dropout, attention=attention
                ).to(DEVICE)

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                best_rmse, _, _ = train_model(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, criterion=nn.MSELoss(), device=DEVICE,
                    max_epochs=max_epochs, patience=patience, use_amp=True, save_checkpoints=False
                )

                trial.report(best_rmse, step=split_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                inner_scores.append(best_rmse)

            return float(np.mean(inner_scores))

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        best_params = study.best_params
        wandb_run.config.update(best_params)

        # Final training step: use best params on the full outer train set
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_outer_train, random_state=SEED)
        tr_sub_idx, val_sub_idx = next(gss.split(X_train, y_train, groups=group_train))

        best_model = TimeLSTM(
            input_dim=X_train.shape[-1], time_dim=t_train.shape[-1],
            hidden_size=best_params["hidden_size"], num_layers=best_params["num_layers"],
            dropout=best_params["dropout"], attention=best_params["attention"]
        ).to(DEVICE)

        train_ds_final = WarfarinDataset(X_train[tr_sub_idx], y_train[tr_sub_idx], t_train[tr_sub_idx], 
                                group_ids=group_train[tr_sub_idx], window_size=best_params["window_size"], allow_padding=True)
        val_ds_final = WarfarinDataset(X_train[val_sub_idx], y_train[val_sub_idx], t_train[val_sub_idx], 
                                    group_ids=group_train[val_sub_idx], window_size=best_params["window_size"], allow_padding=True)
        test_ds = WarfarinDataset(X_test, y_test, t_test, group_ids=group_test, 
                                window_size=best_params["window_size"], allow_padding=True)

        train_loader_final = DataLoader(train_ds_final, batch_size=batch_size, shuffle=True,  **LOADER_KW)
        val_loader_final   = DataLoader(val_ds_final,   batch_size=batch_size, shuffle=False, **LOADER_KW)
        test_loader        = DataLoader(test_ds,        batch_size=batch_size, shuffle=False, **LOADER_KW)

        train_model(
            model=best_model, train_loader=train_loader_final, val_loader=val_loader_final,
            optimizer=torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"]),
            criterion=nn.MSELoss(), device=DEVICE, trial_number=f"outer_fold_{outer_fold}",
            max_epochs=max_epochs, patience=patience, wandb_run=wandb_run, output_dir=OUTPUT_DIR,
            use_amp=True, save_checkpoints=True
        )

        # Evaluation on the completely unseen outer-test set
        best_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["features"].to(DEVICE); t = batch["time"].to(DEVICE); yb = batch["label"].to(DEVICE)
                out = best_model(x, t)
                preds.extend(out.flatten().cpu().numpy()); targets.extend(yb.flatten().cpu().numpy())

        preds_arr = np.array(preds); targets_arr = np.array(targets)
        rmse_score = rmse(targets_arr, preds_arr)
        mae_score = float(mean_absolute_error(targets_arr, preds_arr))
        w20_score = pct_within_20pct(targets_arr, preds_arr)

        fold_results.append({"fold": outer_fold, "rmse": rmse_score, "mae": mae_score, "within20": w20_score})
        
        if save_predictions:
            pd.DataFrame({"fold": outer_fold, "y_true": targets_arr, "y_pred": preds_arr}).to_csv(
                os.path.join(OUTPUT_DIR, f"predictions_outer_fold_{outer_fold}.csv"), index=False)

        wandb_run.finish()

    # Reporting the overall performance across all folds
    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        print(results_df)
        print(f"\nAverage RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
        print(f"Average MAE: {results_df['mae'].mean():.4f}")
        print(f"Average Within 20%: {results_df['within20'].mean():.3%}")
        results_df.to_csv(os.path.join(OUTPUT_DIR, "nested_cv_summary_metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    nested_cv_pipeline(
        df_path="warfarin.csv", outer_splits=5, inner_splits=3, n_trials=30,
        bin_quantiles=10, max_epochs=100, patience=10, batch_size=256,
        save_predictions=True, val_frac_outer_train=0.1, specific_fold=args.fold
    )