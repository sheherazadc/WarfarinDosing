import os, gc, json, random, warnings, argparse, joblib
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb

from preprocess import preprocess_data, build_preprocessor, filter_treatment_episodes, NUM_COLS_90, BIN_COLS_90
from train.dataset import WarfarinDataset
from Warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
from train.train import train_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
# These settings ensure we get the maximum performance out of the GPU
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN = (DEVICE.type == "cuda")
LOADER_KW = dict(num_workers=2, pin_memory=PIN, persistent_workers=False, prefetch_factor=2)

MODEL_NAME = "NET"
BASELINE_DIR     = "NET_outputs_nested_cv_baseline"  
FULL_OUTPUT_DIR  = "NET_outputs_nested_cv"

os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", os.path.abspath(FULL_OUTPUT_DIR))
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

TARGET = "inr"
TIME_COLS = [ 'treatment_day',]
DT_INDEX = 0 
# These are the components we will systematically disable
ABLATION_CHOICES = ["none", "no_featatt", "no_timeatt", "no_attention", "no_decay"]

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def pct_within_20pct(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom) <= 0.20))

def _wandb_init_compat(**kw):
    try: return wandb.init(finish_previous=True, **kw)
    except TypeError: return wandb.init(**kw)

class NoFeatureAttention(nn.Module):
    """
    A 'fake' attention module. It returns the features exactly as they are, 
    effectively telling the model to treat every variable as equally important.
    """
    def __init__(self, input_dim): 
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x):
        B, T, F = x.shape
        # We assign an equal 1/F weight to every feature
        uniform_attention = torch.ones(B, F, device=x.device, dtype=x.dtype) / F
        return x, uniform_attention

class NoTimeAttention(nn.Module):
    """
    Disables the temporal summary. Instead of weighing all visits, 
    the model is forced to only look at the most recent hospital visit.
    """
    def forward(self, h, time_intervals, mask=None):
        if mask is not None:
            # We locate the last non-padded step in the patient's history
            lengths = mask.sum(dim=1).clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(h.size(0), 1, h.size(-1))
            last_hidden = h.gather(1, idx).squeeze(1)
        else:
            last_hidden = h[:, -1, :]
    
        return last_hidden, None

def monkey_patch_ablation_modules(model, ablation: str, input_dim: int):
    """
    A runtime surgery on the model. We swap out standard components 
    with the 'No-Op' modules defined above based on the experiment type.
    """
    if ablation == "no_featatt":
        model.feature_attention = NoFeatureAttention(input_dim)
    elif ablation == "no_timeatt":
        model.time_attention = NoTimeAttention()
    
    return model

def prepare_ablated_data(X_train, X_test, t_train, t_test, ablation: str):
    """
    Prepares the data for specific tests. For example, in the 'no_decay' test, 
    we zero out all time gaps so the model can't tell how far apart visits were.
    """
    Xtr, Xte = X_train.copy(), X_test.copy()
    ttr, tte = t_train.copy(), t_test.copy()
    if ablation == "no_decay":
        ttr[...] = 0.0; tte[...] = 0.0
    return Xtr, Xte, ttr, tte

def nested_cv_pipeline(
    df_path: str,
    ablation: str = "none",
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_trials: int = 20,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    val_frac_outer_train: float = 0.1,
    specific_fold: int = None,
):
    """
    Main loop that handles the nested cross-validation for a specific ablated model version.
    """
    # Create separate folders for each experiment to avoid overwriting data
    out_dir = BASELINE_DIR if ablation == "none" else f"{FULL_OUTPUT_DIR}_{ablation}"
    os.makedirs(out_dir, exist_ok=True)
    ATTN_SUBDIR = os.path.join(out_dir, MODEL_NAME)
    os.makedirs(ATTN_SUBDIR, exist_ok=True)

    os.environ["WANDB_DIR"] = os.path.abspath(out_dir)

    print(f"[INFO] Output dir: {out_dir} | Ablation: {ablation}")

    df_raw = pd.read_csv(df_path)
    df = preprocess_data(df_raw)
    df = filter_treatment_episodes(
        df, patient_id_col='subject_id', date_col='inr_datetime',    
        max_gap_days=100, inr_col='inr'
    )

    drop_cols = ['subject_id', TARGET] + TIME_COLS
    X_df_full = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    X_time_full = df[TIME_COLS].values.astype(np.float32)
    y_full = df[TARGET].values.astype(np.float32)
    groups_full = df['subject_id'].values
    y_bins_full = pd.qcut(y_full, q=10, labels=False, duplicates='drop')

    outer_cv = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    fold_results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_df_full, y_bins_full, groups_full), start=1):
        if specific_fold is not None and outer_fold != specific_fold:
            continue

        print(f"\nOuter Fold {outer_fold}/{outer_splits}")

        preprocessor_outer = build_preprocessor(NUM_COLS_90, BIN_COLS_90)
        X_train = preprocessor_outer.fit_transform(X_df_full.iloc[train_idx])
        X_test  = preprocessor_outer.transform(X_df_full.iloc[test_idx])

        prep_path = os.path.join(out_dir, f"net_preprocessor_outer_fold_{outer_fold}.pkl")
        joblib.dump(preprocessor_outer, prep_path)

        y_train = y_full[train_idx]; y_test = y_full[test_idx]
        t_train = X_time_full[train_idx]; t_test = X_time_full[test_idx]
        g_train = groups_full[train_idx]; g_test = groups_full[test_idx]

        # Inner folds are pre-processed here to speed up the hyperparameter search
        y_bins_train = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
        inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        X_df_outer_train = X_df_full.iloc[train_idx]

        precomputed = []
        for k, (tr_pos, va_pos) in enumerate(inner_cv.split(np.zeros_like(y_train), y_bins_train, groups=g_train)):
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

        # Optuna handles the hyperparameter tuning for each ablated version
        pruner = MedianPruner(n_warmup_steps=2)
        study = optuna.create_study(direction="minimize", pruner=pruner, sampler=TPESampler(seed=SEED))

        def objective(trial: optuna.trial.Trial):
            window_size   = trial.suggest_int("window_size", 3, 20)
            hidden_size   = trial.suggest_int("hidden_size", 16, 256, log=True)  
            num_layers    = trial.suggest_int("num_layers", 2, 5)
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) 
            dropout       = trial.suggest_float("dropout", 0.1, 0.5)
            
            # Logic to force attention on/off based on the experiment name
            if ablation == "no_attention":
                attention = False
            else:
                attention = True if ablation != "none" else trial.suggest_categorical("attention", [False, True])

            scores = []
            for s, pack in enumerate(precomputed):
                # Apply data-level ablations (like zeroing time)
                X_tr_a, X_va_a, t_tr_a, t_va_a = prepare_ablated_data(pack["X_tr"], pack["X_val"], pack["t_tr"], pack["t_val"], ablation)

                train_ds = WarfarinDataset(X_tr_a, pack["y_tr"], t_tr_a, group_ids=pack["g_tr"], 
                                         window_size=window_size, allow_padding=True)
                val_ds   = WarfarinDataset(X_va_a, pack["y_val"], t_va_a, group_ids=pack["g_val"], 
                                         window_size=window_size, allow_padding=True)

                train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **LOADER_KW)
                val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **LOADER_KW)

                model = NETLSTM(
                    input_dim=X_tr_a.shape[-1], time_dim=t_tr_a.shape[-1],
                    hidden_size=hidden_size, num_layers=num_layers,
                    dropout=dropout, attention=attention, use_pack=True, dt_index=DT_INDEX,
                ).to(DEVICE)
                
                # Apply module-level ablations (identity swaps)
                model = monkey_patch_ablation_modules(model, ablation, X_tr_a.shape[-1])

                opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
                
                best_rmse, _, _ = train_model(
                    model=model, train_loader=train_ld, val_loader=val_ld,
                    optimizer=opt, criterion=nn.MSELoss(), device=DEVICE,
                    trial_number=f"{ablation}_fold{outer_fold}_trial{trial.number}_split{s}",
                    max_epochs=max_epochs, patience=patience, use_amp=True, save_checkpoints=False,
                )
                scores.append(best_rmse)
                del model, opt, train_ld, val_ld, train_ds, val_ds
                torch.cuda.empty_cache(); gc.collect()

            return float(np.mean(scores))

        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        best = study.best_params

        # After finding the best settings, retrain the final ablated model for this fold
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_outer_train, random_state=SEED)
        tr_sub, va_sub = next(gss.split(X_train, y_train, groups=g_train))

        X_train_a, X_test_a, t_train_a, t_test_a = prepare_ablated_data(X_train, X_test, t_train, t_test, ablation)

        train_ds_final = WarfarinDataset(X_train_a[tr_sub], y_train[tr_sub], t_train_a[tr_sub], 
                                       group_ids=g_train[tr_sub], window_size=int(best["window_size"]), allow_padding=True)
        val_ds_final   = WarfarinDataset(X_train_a[va_sub], y_train[va_sub], t_train_a[va_sub], 
                                       group_ids=g_train[va_sub], window_size=int(best["window_size"]), allow_padding=True)
        test_ds        = WarfarinDataset(X_test_a, y_test, t_test_a, group_ids=g_test, 
                                       window_size=int(best["window_size"]), allow_padding=True)

        train_ld = DataLoader(train_ds_final, batch_size=batch_size, shuffle=True,  **LOADER_KW)
        val_ld   = DataLoader(val_ds_final,   batch_size=batch_size, shuffle=False, **LOADER_KW)
        test_ld  = DataLoader(test_ds,        batch_size=batch_size, shuffle=False, **LOADER_KW)

        att_flag = bool(best.get("attention", True))
        if ablation == "no_attention": att_flag = False

        model = NETLSTM(
            input_dim=X_train_a.shape[-1], time_dim=t_train_a.shape[-1],
            hidden_size=int(best["hidden_size"]), num_layers=int(best["num_layers"]),
            dropout=float(best["dropout"]), attention=att_flag, use_pack=True, dt_index=DT_INDEX,
        ).to(DEVICE)
        model = monkey_patch_ablation_modules(model, ablation, X_train_a.shape[-1])

        run = _wandb_init_compat(project=(f"warfarinNET-ablate-{ablation}"),
                                 name=f"{ablation}_outer_fold_{outer_fold}",
                                 config={"ablation": ablation, **best}, reinit=True)

        train_model(
            model=model, train_loader=train_ld, val_loader=val_ld,
            optimizer=torch.optim.Adam(model.parameters(), lr=float(best["learning_rate"]), weight_decay=1e-4),
            criterion=nn.MSELoss(), device=DEVICE, trial_number=f"{ablation}_outer_fold_{outer_fold}",
            max_epochs=max_epochs, patience=patience, wandb_run=run, output_dir=out_dir, 
            model_name=MODEL_NAME, use_amp=True, save_checkpoints=True
        )

        # Final evaluation on the unseen test set for this fold
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_ld:
                x, t, yb = batch["features"].to(DEVICE), batch["time"].to(DEVICE), batch["label"].to(DEVICE)
                out = model(x, t)
                if isinstance(out, (tuple, list)): out = out[0]
                preds.extend(out.detach().cpu().numpy()); targets.extend(yb.detach().cpu().numpy())

        preds_arr = np.asarray(preds); targets_arr = np.asarray(targets)
        rmse_score = rmse(targets_arr, preds_arr)
        mae_score  = float(mean_absolute_error(targets_arr, preds_arr))
        
        print(f"[Fold {outer_fold}] Ablation: {ablation} | Test RMSE: {rmse_score:.4f} | MAE: {mae_score:.4f}")
        
        # Save results for future statistical comparisons
        pd.DataFrame({"y_true": targets_arr.astype(float), "y_pred": preds_arr.astype(float)}).to_csv(
            os.path.join(out_dir, f"{MODEL_NAME}_predictions_outer_fold_{outer_fold}.csv"), index=False)

        run.finish()
        torch.cuda.empty_cache(); gc.collect()

        fold_results.append({"fold": outer_fold, "rmse": rmse_score, "mae": mae_score})

    results_df = pd.DataFrame(fold_results)
    if not results_df.empty:
        results_df.to_csv(os.path.join(out_dir, f"{MODEL_NAME}_nested_cv_summary_metrics.csv"), index=False)


def parse_ablation_arg(s: str):
    """Simple parser to allow running multiple ablations from a single command line call."""
    s = (s or "none").strip().lower()
    if s == "all": return ABLATION_CHOICES
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or ["none"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", type=str, default="warfarin.csv")
    ap.add_argument("--ablation", type=str, default="none", help="Which part to remove.")
    args = ap.parse_args()

    for abl in parse_ablation_arg(args.ablation):
        nested_cv_pipeline(df_path=args.df, ablation=abl)