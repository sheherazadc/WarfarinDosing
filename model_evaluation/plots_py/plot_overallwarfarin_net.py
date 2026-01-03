import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import torch
import joblib
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
from dataset import WarfarinDataset
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from typing import Dict, List

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def prepare_error_data(df):
    """
    Groups raw performance data into clinical buckets.
    This lets us check if the model's error rate shifts significantly 
    for elderly patients or those on specific dosing ranges.
    """
    df = df.copy()
    df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 30, 50, 65, 80, 120],
            labels=["<30", "30–50", "50–65", "65–80", "80+"]
        )
    if "dose" in df.columns:
        df["dose_group"] = pd.cut(
            df["dose"],  bins=[0, 1, 2, 3.5, 5, float('inf')],
            labels=['(0-1mg)', '(1-2mg)','(2-3.5mg)', '(3.5-5mg)', '(>5mg)'],
        )
    return df

def rmse_compat(y_true, y_pred):
    """Standard root mean square error calculation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def barplot_mae(df, group_col, title, save_path):
    """
    Creates a bar chart of Mean Absolute Error.
    The whiskers represent the standard error, helping us see if differences 
    between groups are likely due to small sample sizes.
    """
    grouped = df.groupby(group_col)['abs_error'].agg(['mean', 'std', 'count']).reset_index()

    plt.figure(figsize=(7, 5))
    plt.bar(grouped[group_col].astype(str),
            grouped['mean'],
            yerr=grouped['std']/np.sqrt(grouped['count']),
            capsize=5,
            alpha=0.75,
            color='orchid')

    plt.title(title)
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.xlabel(group_col.replace("_", " ").title())
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved MAE bar plot: {save_path}")
    return grouped

def calculate_calibration_metrics(y_true, y_pred, n_bins=20):
    """
    Calibration check: Does a predicted INR of 2.5 actually mean the patient 
    is at 2.5? We use binned means and linear regression to check for 
    systemic over or under-estimation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    intercept = float(lr.intercept_)
    slope = float(lr.coef_[0])

    bin_boundaries = np.linspace(y_pred.min(), y_pred.max(), n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_means_pred, bin_means_true, bin_counts = [], [], []

    for i in range(n_bins):
        in_bin = (y_pred >= bin_lowers[i]) & (y_pred < bin_uppers[i])
        if i == n_bins - 1:
            in_bin = (y_pred >= bin_lowers[i]) & (y_pred <= bin_uppers[i])

        if np.sum(in_bin) > 0:
            bin_means_pred.append(float(np.mean(y_pred[in_bin])))
            bin_means_true.append(float(np.mean(y_true[in_bin])))
            bin_counts.append(int(np.sum(in_bin)))
        else:
            bin_means_pred.append(np.nan)
            bin_means_true.append(np.nan)
            bin_counts.append(0)

    bin_means_pred = np.asarray(bin_means_pred, dtype=float)
    bin_means_true = np.asarray(bin_means_true, dtype=float)
    bin_counts = np.asarray(bin_counts, dtype=int)

    valid = ~np.isnan(bin_means_pred)
    bin_means_pred = bin_means_pred[valid]
    bin_means_true = bin_means_true[valid]
    bin_counts = bin_counts[valid]

    if bin_means_pred.size == 0:
        mace = np.nan
        rmsce = np.nan
    else:
        diffs = bin_means_true - bin_means_pred
        w = np.maximum(bin_counts.astype(float), 1.0)
        mace = float(np.sum(np.abs(diffs) * w) / np.sum(w))
        rmsce = float(np.sqrt(np.sum((diffs**2) * w) / np.sum(w)))

    return {
        "intercept": intercept,
        "slope": slope,
        "mace": mace,
        "rmsce": rmsce,
        "bin_means_pred": bin_means_pred,
        "bin_means_true": bin_means_true,
        "bin_counts": bin_counts,
    }

class WarfarinNETAnalyzer:
    """
    Master analyzer for the model. 
    It pulls results from nested cross-validation and generates 
    the visualizations needed for clinical validation.
    """
    
    def __init__(self, output_dir="real_NET_outputs_nested_cv", model_dirs=None, baseline_name=None):
        self.output_dir = output_dir
        self.results_df = None
        self.attention_data = {}
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = "real_BIG_NET_analysis_plots"
        os.makedirs(self.save_dir, exist_ok=True)

        if model_dirs is None:
            model_dirs = {"NET": output_dir}
        self.model_dirs = model_dirs
        self.baseline_name = baseline_name if baseline_name is not None else list(model_dirs.keys())[0]
        
        self.feature_names_18 = [
            'previous_inr', 'alt', 'dose', 'weight_kg', 'bilirubin', 'height_cm',
            'hemoglobin', 'platelet', 'age', 'creatinine', 'previous_dose', 'bmi',
            'on_cyp2c9_inhibitor', 'on_cyp2c9_inducer', 'on_cyp3a4_inducer',
            'on_cyp3a4_inhibitor', 'cardiac_history', 'ethnicity_asian',
            'previous_inr_timediff_hours', 'dose_diff_hours',
        ]
        
        self.time_names_3 = ['treatment_day']

        self._load_results()
        self._load_attention_data()
        self._load_models()

        self._all_model_results = self._load_all_model_results(self.model_dirs)

        try:
            from warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
            self.NETLSTM = NETLSTM
        except ImportError:
            print("Warning: Could not import NET model class")
            self.NETLSTM = None
    
    def _load_results(self):
        """Finds and combines all fold prediction CSVs into one master list."""
        pred_files = glob.glob(os.path.join(self.output_dir, "NET_predictions_outer_fold_*.csv"))
        dfs = []
        for file in pred_files:
            df = pd.read_csv(file)
            if "fold" not in df.columns:
                try:
                    fold_id = int(os.path.basename(file).split("_")[-1].replace(".csv", ""))
                    df["fold"] = fold_id
                except Exception:
                    pass
            dfs.append(df)
        if dfs:
            self.results_df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(pred_files)} prediction files with {len(self.results_df)} total predictions")
        else:
            print("No prediction files found in", self.output_dir)
    
    def _load_attention_data(self):
        """Loads the .npz files containing the time and feature importance scores."""
        attn_files = []
        for sub in ["NET", "WarfarinNet", "WarfarinNET"]:
            attn_files.extend(
                glob.glob(os.path.join(self.output_dir, sub, "attn_outputs_trial_*_best.npz"))
            )

        for file in attn_files:
            try:
                data = np.load(file, allow_pickle=True)
                try:
                    fold_id = os.path.basename(file).split('_')[3]
                except Exception:
                    fold_id = os.path.splitext(os.path.basename(file))[0]
                self.attention_data[fold_id] = {
                    'preds': data.get('preds', None),
                    'targets': data.get('targets', None),
                    'time_attn': data.get('attn_time', None),
                    'feat_attn': data.get('attn_feat', None)
                }
                print(f"Loaded attention data for fold {fold_id}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
            
    def _load_models(self):
        """Reconstructs the trained WarfarinNet models and links them to their specific feature names."""
        model_files = sorted(glob.glob(os.path.join(self.output_dir, "NET_best_model_outer_fold_*.pt")))
        for file in model_files:
            try:
                fold_id = os.path.basename(file).split('_')[-1].replace('.pt', '')
                checkpoint = torch.load(file, map_location=self.device)
                config = checkpoint['config']

                model_kwargs = {
                    'input_dim': config['input_dim'],
                    'time_dim': config['time_dim'],
                    'hidden_size': config['hidden_size'],
                    'num_layers': config['num_layers'],
                    'dropout': config['dropout'],
                    'attention': config.get('attention', True)
                }
                
                if 'decay_mode' in config:
                    model_kwargs['decay_mode'] = config['decay_mode']
                if 'use_skip' in config:
                    model_kwargs['use_skip'] = config['use_skip']

                model = NETLSTM(**model_kwargs)
                
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    if "skip.fc" in str(e):
                        model_kwargs['use_skip'] = False
                        model = NETLSTM(**model_kwargs)
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        raise e
                
                model.eval()

                feat_names = None
                prep_path = config.get('preprocessor_path')
                if prep_path and os.path.exists(prep_path):
                    try:
                        preproc = joblib.load(prep_path)
                        
                        if hasattr(preproc, "get_feature_names_out"):
                            try:
                                raw_names = preproc.get_feature_names_out()
                            except Exception:
                                raw_names = None
                        elif hasattr(preproc, "named_steps"):
                            try:
                                cols_tf = preproc.named_steps.get('cols', None)
                                if cols_tf and hasattr(cols_tf, "get_feature_names_out"):
                                    raw_names = cols_tf.get_feature_names_out()
                                else:
                                    raw_names = None
                            except Exception:
                                try:
                                    steps = list(preproc.named_steps.items())
                                    if len(steps) > 1:
                                        filtered_steps = [(name, step) for name, step in steps 
                                                        if not any(x in name.lower() for x in ['winsor', 'scaler', 'final'])]
                                        if filtered_steps:
                                            partial_pipe = Pipeline(filtered_steps)
                                            n_features = config['input_dim']
                                            dummy_data = pd.DataFrame(np.zeros((1, n_features)))
                                            try:
                                                partial_pipe.fit(dummy_data)
                                                raw_names = partial_pipe.get_feature_names_out()
                                            except:
                                                raw_names = None
                                        else:
                                            raw_names = None
                                    else:
                                        raw_names = None
                                except Exception:
                                    raw_names = None
                        else:
                            raw_names = None
                        
                        if raw_names is not None:
                            feat_names = [str(n).split('__', 1)[-1] if '__' in str(n) else str(n) 
                                        for n in raw_names]
                        
                    except Exception as e:
                        print(f"Fold {fold_id}: could not load preprocessor from {prep_path}: {e}")

                if feat_names is None:
                    expected_dim = config.get('input_dim', len(self.feature_names_18))
                    if expected_dim == 18:
                        feat_names = self.feature_names_18.copy()
                    elif expected_dim == len(self.feature_names_18):
                        feat_names = self.feature_names_18.copy()
                    else:
                        feat_names = [f"feature_{i}" for i in range(expected_dim)]

                self.models[fold_id] = {
                    'model': model,
                    'config': config,
                    'feature_names': feat_names,
                }
                fn_info = len(feat_names) if isinstance(feat_names, (list, tuple)) else "unknown"
                print(f"Loaded model for fold {fold_id} (feature_names={fn_info})")

            except Exception as e:
                print(f"Error loading model {file}: {e}")

    def _load_all_model_results(self, model_dirs):
        """Collects prediction CSVs from every model type for side-by-side comparison."""
        all_results = {}
        for name, mdir in model_dirs.items():
            files = glob.glob(os.path.join(mdir, "*_predictions_outer_fold_*.csv")) + \
                    glob.glob(os.path.join(mdir, "*predictions_outer_fold_*.csv")) + \
                    glob.glob(os.path.join(mdir, "NET_predictions_outer_fold_*.csv"))
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    if "fold" not in df.columns:
                        try:
                            fold_id = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
                            df["fold"] = fold_id
                        except Exception:
                            pass
                    df["model"] = name
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read {f}: {e}")
            if dfs:
                all_results[name] = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(dfs)} prediction files for model {name}")
            else:
                print(f"No prediction files found for model {name} in {mdir}")
        return all_results

    @staticmethod
    def _fold_metrics(df):
        """Calculates standard error metrics per cross-validation fold."""
        by_fold = df.groupby("fold").apply(
            lambda d: pd.Series({
                "mae": mean_absolute_error(d["y_true"], d["y_pred"]),
                "rmse": rmse_compat(d["y_true"], d["y_pred"]),
                "n": len(d)
            })
        ).reset_index()
        return by_fold

    @staticmethod
    def _calibration_table(df, n_bins=20, min_bin=10):
        """Checks if predicted scores match actual outcome frequencies in buckets."""
        x = np.asarray(df["y_pred"], dtype=float)
        y = np.asarray(df["y_true"], dtype=float)
        if len(x) < 3:
            return None
        slope, intercept = np.polyfit(x, y, 1)

        try:
            bins = pd.qcut(x, q=n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(x, bins=n_bins)

        tab = df.groupby(bins).agg(y_true_mean=("y_true", "mean"),
                                   y_pred_mean=("y_pred", "mean"),
                                   n=("y_true", "count")).dropna()
        tab = tab[tab["n"] >= min_bin]
        if tab.empty:
            mace = np.nan
            rmsce = np.nan
        else:
            diffs = (tab["y_true_mean"] - tab["y_pred_mean"]).values
            mace = np.mean(np.abs(diffs))
            rmsce = float(np.sqrt(np.mean(diffs**2)))

        return pd.Series({
            "cal_intercept": float(intercept),
            "cal_slope": float(slope),
            "MACE": float(mace) if np.isfinite(mace) else np.nan,
            "RMSCE": float(rmsce) if np.isfinite(rmsce) else np.nan
        })

    def build_calibration_results_equalwidth(self, n_bins=20, models=None):
        """Aggregates calibration data using equal-width bins for smoother plotting."""
        cal = {}
        target_models = models or list(self._all_model_results.keys())
        for name in target_models:
            dfm = self._all_model_results.get(name)
            if dfm is None or dfm.empty or "y_true" not in dfm or "y_pred" not in dfm:
                continue

            fold_rows = []
            all_y_true, all_y_pred = [], []

            if "fold" in dfm.columns:
                for f, d in dfm.groupby("fold"):
                    m = calculate_calibration_metrics(d["y_true"].to_numpy(),
                                                      d["y_pred"].to_numpy(),
                                                      n_bins=n_bins)
                    fold_rows.append({
                        "fold": int(f),
                        "bin_means_pred": m["bin_means_pred"],
                        "bin_means_true": m["bin_means_true"],
                        "bin_counts": m["bin_counts"],
                    })
                    all_y_true.extend(d["y_true"].to_numpy())
                    all_y_pred.extend(d["y_pred"].to_numpy())

            all_y_true = np.asarray(all_y_true, dtype=float)
            all_y_pred = np.asarray(all_y_pred, dtype=float)

            cal[name] = {
                "fold_calibration_data": fold_rows,
                "all_y_true": all_y_true,
                "all_y_pred": all_y_pred,
            }
        return cal


    @staticmethod
    def _paired_test_vs_baseline(baseline, challenger):
        """Statistical test to see if performance improvements are likely real or random noise."""
        merged = pd.merge(baseline, challenger, on="fold", suffixes=("_base", "_chall"))
        out = {}
        for metric in ["mae", "rmse"]:
            d = merged[f"{metric}_chall"] - merged[f"{metric}_base"]
            d = d.dropna()
            n = len(d)
            if n < 2:
                out[metric] = {"test": None}
                continue
            try:
                sh_p = stats.shapiro(d).pvalue
            except Exception:
                sh_p = 0.0
            if sh_p >= 0.05:
                t_res = stats.ttest_rel(merged[f"{metric}_chall"], merged[f"{metric}_base"], nan_policy="omit")
                mean_diff = float(np.mean(d))
                sd = float(np.std(d, ddof=1))
                se = sd / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=n-1)
                ci_low = mean_diff - t_crit * se
                ci_high = mean_diff + t_crit * se
                dz = mean_diff / sd if sd > 0 else np.nan
                out[metric] = {
                    "test": "paired_t",
                    "n": n,
                    "pvalue": float(t_res.pvalue),
                    "mean_diff": mean_diff,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "effect": "cohen_dz",
                    "effect_size": float(dz)
                }
            else:
                try:
                    w = stats.wilcoxon(merged[f"{metric}_chall"], merged[f"{metric}_base"])
                    p = float(w.pvalue)
                    z = stats.norm.isf(p/2.0)
                    z = z * np.sign(np.mean(d))
                    r = z / np.sqrt(n)
                except Exception:
                    p, r = np.nan, np.nan
                mean_diff = float(np.mean(d))
                boot = []
                rng = np.random.RandomState(13)
                for _ in range(2000):
                    s = rng.choice(d, size=n, replace=True)
                    boot.append(np.mean(s))
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                out[metric] = {
                    "test": "wilcoxon",
                    "n": n,
                    "pvalue": p,
                    "mean_diff": mean_diff,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "effect": "r",
                    "effect_size": float(r)
                }
        return out


    @staticmethod
    def _calc_calibration_bins(y_true, y_pred, n_bins=20, use_quantiles=True):
        """Sorts predictions into bins to calculate observed versus expected results."""
        x = np.asarray(y_pred, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if use_quantiles:
            try:
                bins = pd.qcut(x, q=n_bins, duplicates="drop")
            except Exception:
                bins = pd.cut(x, bins=n_bins)
        else:
            bins = pd.cut(x, bins=n_bins)

        tab = (pd.DataFrame({"x": x, "y": y, "bin": bins})
                 .groupby("bin", observed=True)
                 .agg(bin_means_pred=("x", "mean"),
                      bin_means_true=("y", "mean"),
                      n=("y", "count"))
                 .dropna())
        return {
            "bin_means_pred": tab["bin_means_pred"].to_numpy(),
            "bin_means_true": tab["bin_means_true"].to_numpy(),
            "n": tab["n"].to_numpy(),
        }

    def build_calibration_results(self, n_bins=20, models=None):
        """Constructs a dictionary of all calibration metadata needed for plotting."""
        cal = {}
        target_models = models or list(self._all_model_results.keys())
        for name in target_models:
            dfm = self._all_model_results.get(name)
            if dfm is None or dfm.empty or "y_true" not in dfm or "y_pred" not in dfm:
                continue

            all_y_true = dfm["y_true"].to_numpy(dtype=float)
            all_y_pred = dfm["y_pred"].to_numpy(dtype=float)

            fold_rows = []
            if "fold" in dfm.columns:
                for f, d in dfm.groupby("fold"):
                    bins = self._calc_calibration_bins(d["y_true"], d["y_pred"], n_bins=n_bins)
                    fold_rows.append({
                        "fold": int(f),
                        "bin_means_pred": bins["bin_means_pred"],
                        "bin_means_true": bins["bin_means_true"],
                        "n": bins["n"]
                    })
            cal[name] = {
                "fold_calibration_data": fold_rows,
                "all_y_true": all_y_true,
                "all_y_pred": all_y_pred
            }
        return cal
    
    def plot_calibration_curves_per_fold(self, calibration_results, output_dir="figures_calibration", top_n=3, n_bins=10):
        """Draws the reliability diagram showing how well predicted values track reality."""
        os.makedirs(output_dir, exist_ok=True)

        model_names = list(calibration_results.keys())[:top_n]
        if not model_names:
            print("No models to plot.")
            return

        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]

        fold_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

        for i, model_name in enumerate(model_names):
            results = calibration_results[model_name]
            fold_calibration_data = results['fold_calibration_data']
            all_y_true = np.asarray(results['all_y_true'], dtype=float)
            all_y_pred = np.asarray(results['all_y_pred'], dtype=float)
            ax = axes[i]

            for fd in fold_calibration_data:
                x = np.asarray(fd['bin_means_pred'], dtype=float)
                y = np.asarray(fd['bin_means_true'], dtype=float)
                if x.size == 0 or y.size == 0:
                    continue
                o = np.argsort(x)
                x, y = x[o], y[o]
                c = fold_colors[(fd['fold'] - 1) % len(fold_colors)]
                ax.plot(x, y, '-', lw=0.9, alpha=0.7, color=c,
                        label=f'Fold {fd["fold"]}' if i == 0 else None)
                ax.plot(x, y, 'o', ms=2, alpha=0.8, color=c)

            overall = calculate_calibration_metrics(all_y_true, all_y_pred, n_bins=n_bins)
            x_avg = np.asarray(overall['bin_means_pred'], dtype=float)
            y_avg = np.asarray(overall['bin_means_true'], dtype=float)
            if x_avg.size and y_avg.size:
                o = np.argsort(x_avg)
                x_avg, y_avg = x_avg[o], y_avg[o]
                ax.plot(x_avg, y_avg, '-', lw=2, color='crimson',
                        label='Average' if i == 0 else None)
                ax.plot(x_avg, y_avg, 'o', ms=3, color='crimson')

            lo = float(min(all_y_true.min(), all_y_pred.min()))
            hi = float(max(all_y_true.max(), all_y_pred.max()))
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0,
                    label='Perfect calibration' if i == 0 else None)

            try:
                lr = LinearRegression().fit(all_y_pred.reshape(-1, 1), all_y_true)
                xs = np.linspace(lo, hi, 100)
                ax.plot(xs, lr.predict(xs.reshape(-1, 1)), color='gray', lw=1.5,
                        label=(f'Calibration line (slope={lr.coef_[0]:.2f})' if i == 0 else None))
            except Exception:
                pass

            ax.set_xlim(lo, 6)
            ax.set_ylim(lo, 6.5)
            ax.set_xlabel('Mean Predicted INR')
            ax.set_ylabel('Mean Observed INR')
            ax.set_title(f'{model_name}\nCalibration Curve')

            if i == 0:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            ax.grid(True, alpha=0.3)

        out_path = os.path.join(output_dir, 'calibration_curves_per_fold.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

    def load_xgboost_kde_data(self, xgb_predictions_path="figures_xgb/xgboost_predictions.csv"):
        """Imports XGBoost predictions for distribution shape comparison."""
        if os.path.exists(xgb_predictions_path):
            try:
                xgb_df = pd.read_csv(xgb_predictions_path)
                print(f"Loaded XGBoost predictions: {len(xgb_df)} samples")
                return xgb_df
            except Exception as e:
                print(f"Error loading XGBoost predictions: {e}")
                return None
        else:
            print(f"XGBoost predictions not found at: {xgb_predictions_path}")
            return None

    def plot_kde_comparison_three_way(self, xgb_predictions_path="figures_xgb/xgboost_predictions.csv", save_path=None):
        """Draws a smoothed probability density plot for Actual, LSTM, and XGBoost results."""
        if self.results_df is None:
            return
        
        xgb_df = self.load_xgboost_kde_data(xgb_predictions_path)
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=self.results_df['y_true'], label='Actual INR', linestyle='--', color='black', alpha=0.8, linewidth=2.5)
        sns.kdeplot(data=self.results_df['y_pred'], label='WarfarinLSTM', color='orchid', alpha=0.7, linewidth=2)
        
        if xgb_df is not None and 'pred_inr' in xgb_df.columns:
            sns.kdeplot(data=xgb_df['pred_inr'], label='XGBoost', color='mediumslateblue', alpha=0.7, linewidth=2)
        
        plt.xlabel('INR Value')
        plt.ylabel('Density')
        plt.title('Distribution Comparison: Actual vs Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        actual_mean = self.results_df['y_true'].mean()
        net_mean = self.results_df['y_pred'].mean()
        stats_text = f'Actual Mean: {actual_mean:.3f}\nWarfarinLSTM Mean: {net_mean:.3f}'
        
        if xgb_df is not None and 'pred_inr' in xgb_df.columns:
            stats_text += f'\nXGBoost Mean: {xgb_df["pred_inr"].mean():.3f}'
        
        plt.text(0.83, 0.25, stats_text, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    
    def plot_net_vs_xgb_subgroup_comparison(self, xgb_csv_dir="figures_xgb_cvsubgroups", save_dir=None):
        """Generates side-by-side bar charts comparing WarfarinNet versus XGBoost across demographic slices."""
        if save_dir is None:
            save_dir = self.save_dir

        xgb_data = self.load_xgboost_subgroup_csvs(xgb_csv_dir)
        if not xgb_data:
            return

        if self.results_df is None:
            return

        net_df = prepare_error_data(self.results_df)

        column_mapping = {
            "sex": "gender",
            "age_group": "age_group",
            "dose_group": "dose_group",
            "ethnicity_asian": "ethnicity_asian",
            "race": "ethnicity",
            "ethnicity": "ethnicity",      
        }

        def _norm_eth(x: str) -> str:
            if pd.isna(x):
                return x
            s = str(x).strip()
            s = s.replace(" / ", "/")
            s = s.replace("Hispanic/Latino", "Hispanic / Latino")
            s = s.replace("Black/African American", "Black / African American")
            s = s.replace("Caucasian", "White")
            return s

        for group_col, xgb_df in xgb_data.items():
            net_col = column_mapping.get(group_col)

            if net_col is None or net_col not in net_df.columns:
                continue

            net_tmp = net_df.dropna(subset=[net_col]).copy()

            if net_col in ("ethnicity", "race"):
                net_tmp[net_col] = net_tmp[net_col].map(_norm_eth)

            net_grouped = (net_tmp
                .groupby(net_col, observed=True)['abs_error']
                .agg(['mean', 'std', 'count'])
                .reset_index())
            net_grouped['MAE'] = net_grouped['mean']
            net_grouped['MAE_se'] = net_grouped['std'] / np.sqrt(net_grouped['count'])
            net_grouped[group_col] = net_grouped[net_col].astype(str)
            net_grouped = net_grouped[[group_col, 'MAE', 'MAE_se', 'count']]

            xgb_df_clean = xgb_df.copy()
            xgb_df_clean[group_col] = xgb_df_clean[group_col].astype(str)
            if group_col in ("ethnicity", "race"):
                xgb_df_clean[group_col] = xgb_df_clean[group_col].map(_norm_eth)

            count_col = 'n_total' if 'n_total' in xgb_df_clean.columns else 'n_mean'

            comparison_df = pd.merge(
                net_grouped,
                xgb_df_clean[[group_col, 'MAE_mean', 'MAE_sd', count_col]],
                on=group_col, how='inner', suffixes=('_NET', '_XGB')
            )
            if comparison_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))
            x_pos = np.arange(len(comparison_df))
            width = 0.45

            ax.bar(x_pos - width/2, comparison_df['MAE'], width, yerr=comparison_df['MAE_se'], capsize=5, alpha=0.8, color='orchid', label='WarfarinLSTM')
            ax.bar(x_pos + width/2, comparison_df['MAE_mean'], width, yerr=comparison_df['MAE_sd'], capsize=5, alpha=0.8, color='mediumslateblue', label='XGBoost')

            ax.set_xticks(x_pos)

            if group_col in ("race", "ethnicity"):
                def _two_line(lbl: str) -> str:
                    if lbl == "Black / African American": return "Black /\nAfrican American"
                    if lbl == "Hispanic / Latino": return "Hispanic /\nLatino"
                    return lbl
                ax.set_xticklabels([_two_line(v) for v in comparison_df[group_col].astype(str)], fontsize=10)
            else:
                ax.set_xticklabels(comparison_df[group_col].astype(str), fontsize=10)

            ax.set_ylabel('MAE')
            ax.set_title(f'Comparison by {group_col}')
            ax.legend(frameon=False)
            ax.grid(True, axis='y', color='lightgrey', alpha=0.4)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"NET_vs_XGB_comparison_{group_col}.png"), dpi=300)
            plt.close()


    def load_xgboost_subgroup_csvs(self, xgb_csv_dir="figures_xgb_cvsubgroups"):
        """Loads summarized MAE results for XGBoost from pre-saved CSV files."""
        xgb_subgroup_data = {}
        file_map = {
            "XGBoost_subgroup_MAE_sex.csv": "sex",
            "XGBoost_subgroup_MAE_age_group.csv": "age_group",
            "XGBoost_subgroup_MAE_dose_group.csv": "dose_group", 
            "XGBoost_subgroup_MAE_ethnicity_asian.csv": "ethnicity_asian",
            "XGBoost_subgroup_MAE_ethnicity.csv": "ethnicity" 
        }
        
        for filename, group_col in file_map.items():
            csv_path = os.path.join(xgb_csv_dir, filename)
            if os.path.exists(csv_path):
                try:
                    xgb_subgroup_data[group_col] = pd.read_csv(csv_path)
                except Exception:
                    pass
        return xgb_subgroup_data

    def generate_complete_comparison_analysis(self, 
                                            xgb_predictions_path="figures_xgb/xgboost_predictions.csv",
                                            xgb_csv_dir="figures_xgb_cvsubgroups", 
                                            save_dir=None):
        """Main entry point to trigger the full suite of plots for the final report."""
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.plot_kde_comparison_three_way(xgb_predictions_path, os.path.join(save_dir, "three_way_kde_comparison.png"))
        self.plot_net_vs_xgb_subgroup_comparison(xgb_csv_dir, save_dir)
        self.generate_all_plots(save_dir)
        self.generate_paper_figs()
        
    def fig_4_4_foldwise_violin_box(self, save_path=None):
        """Visualizes the variance in error rates across the 5 cross-validation folds."""
        if not self._all_model_results:
            return
        
        fold_tables = []
        for name, df in self._all_model_results.items():
            if "fold" not in df.columns:
                continue
            ft = self._fold_metrics(df)
            ft["model"] = name
            fold_tables.append(ft)
        if not fold_tables:
            return
        fold_all = pd.concat(fold_tables, ignore_index=True)
        m = fold_all.melt(id_vars=["fold", "model"], value_vars=["mae", "rmse"], var_name="metric", value_name="value")

        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(data=m, x="model", y="value", hue="metric", split=True, inner=None, cut=0)
        sns.boxplot(data=m, x="model", y="value", hue="metric", showcaps=True, boxprops={'facecolor':'None'}, showfliers=False, whiskerprops={'linewidth':2})
        plt.title("Fold-wise Error Variance")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

        stats_rows = []
        base_name = self.baseline_name
        if base_name in self._all_model_results:
            base_fold = self._fold_metrics(self._all_model_results[base_name])
            for name, df in self._all_model_results.items():
                if name == base_name: continue
                res = self._paired_test_vs_baseline(base_fold, self._fold_metrics(df))
                for metric, vals in res.items():
                    if vals.get("test"):
                        row = {"baseline": base_name, "model": name, "metric": metric}
                        row.update(vals)
                        stats_rows.append(row)
            if stats_rows:
                pd.DataFrame(stats_rows).to_csv(os.path.join(self.save_dir, "Fig_4_4_stats_vs_baseline.csv"), index=False)

    def table_4_8_calibration_metrics(self, save_csv=None, n_bins=20):
        """Calculates numerical calibration stats like MACE and slope for the final tables."""
        rows = []
        for name, df in self._all_model_results.items():
            if df.empty or "fold" not in df.columns: continue
            for f, d in df.groupby("fold"):
                cal = self._calibration_table(d, n_bins=n_bins)
                if cal is not None:
                    r = {"model": name, "fold": f}
                    r.update(cal.to_dict())
                    rows.append(r)
        if rows:
            tab = pd.DataFrame(rows).sort_values(["model", "fold"])
            if save_csv: tab.to_csv(save_csv, index=False)
            return tab

    def fig_4_5_calibration_curves(self, top_k=3, save_path=None, n_bins=20):
        """Draws specific calibration diagrams for the top performing models."""
        if not self._all_model_results: return
        rmse_rank = []
        for name, df in self._all_model_results.items():
            if df is not None and not df.empty:
                rmse_rank.append((name, rmse_compat(df["y_true"], df["y_pred"])))
        if not rmse_rank: return

        pick = [n for n, _ in sorted(rmse_rank, key=lambda x: x[1])[:max(1, top_k)]]
        plt.figure(figsize=(6*len(pick), 5))
        for i, name in enumerate(pick, 1):
            d = self._all_model_results[name]
            x, y = d["y_pred"].to_numpy(), d["y_true"].to_numpy()
            try: bins = pd.qcut(x, q=n_bins, duplicates="drop")
            except: bins = pd.cut(x, bins=n_bins)
            b = d.groupby(bins).agg(y_true_mean=("y_true", "mean"), y_pred_mean=("y_pred", "mean"), n=("y_true", "count")).dropna()

            ax = plt.subplot(1, len(pick), i)
            ax.plot([b["y_pred_mean"].min(), b["y_pred_mean"].max()], [b["y_pred_mean"].min(), b["y_pred_mean"].max()], '--', alpha=0.7)
            ax.plot(b["y_pred_mean"], b["y_true_mean"], 'o-', lw=2)
            lr = LinearRegression().fit(x.reshape(-1, 1), y)
            ax.text(0.05, 0.95, f"Slope={lr.coef_[0]:.2f}", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.7))
            ax.set_title(f"Calibration: {name}")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()


    def fig_4_6_delta_mae_bar(self, save_path=None):
        """Horizontal bar chart showing the difference in error compared to the baseline."""
        base = self.baseline_name
        if base not in self._all_model_results: return
        base_fold = self._fold_metrics(self._all_model_results[base])[["fold", "mae"]].rename(columns={"mae": "mae_base"})
        rows = []
        for name, df in self._all_model_results.items():
            if name == base: continue
            m = pd.merge(base_fold, self._fold_metrics(df)[["fold", "mae"]].rename(columns={"mae": "mae_ch"}), on="fold")
            d = (m["mae_ch"] - m["mae_base"]).values
            if len(d) < 2: continue
            mean, se = d.mean(), d.std(ddof=1)/np.sqrt(len(d))
            t_crit = stats.t.ppf(0.975, df=len(d)-1)
            rows.append({"model": name, "delta_mae": mean, "ci": t_crit*se})
        if rows:
            df_plot = pd.DataFrame(rows).sort_values("delta_mae")
            plt.figure(figsize=(8, 5))
            plt.barh(df_plot["model"], df_plot["delta_mae"], xerr=df_plot["ci"], capsize=4, alpha=0.8)
            plt.axvline(0, color="k", linestyle="--")
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=300)
            plt.close()

    def fig_4_9_time_attention_heatmap(self, max_patients=10, save_path=None):
        """Heatmap showing where the model prioritized attention across historical patient steps."""
        return self.plot_time_attention_heatmap(max_patients=max_patients, save_path=save_path)

    def table_4_11_feature_attention_summary(self, save_csv=None):
        """Summarizes which clinical features the model found most relevant across all folds."""
        if not self.attention_data: return
        import re
        rows = []
        for fold_id, data in self.attention_data.items():
            A = data.get('feat_attn')
            if A is None or A.ndim < 1: continue
            if A.ndim == 1: A = A[None, :]
            
            fold_key = str(fold_id)
            if fold_key not in self.models:
                m = re.search(r'(\d+)', fold_key)
                if m: fold_key = m.group(1)

            names = self.models.get(fold_key, {}).get('feature_names', self.feature_names_18)
            rows.append(pd.Series(A.mean(axis=0), index=names[:A.shape[1]], name=f"fold_{fold_key}"))

        if rows:
            feat_df = pd.concat(rows, axis=1).T
            summary = pd.DataFrame({'Mean_Weight': feat_df.mean(axis=0), 'Fold_Stability_Std': feat_df.std(axis=0)})
            summary = summary.sort_values('Mean_Weight', ascending=False)
            if save_csv: summary.round(6).to_csv(save_csv)
            return summary

    def _load_actual_feature_names(self):
        """Helper to extract names from binary files if they weren't explicitly provided."""
        for model_file in glob.glob(os.path.join(self.output_dir, "NET_best_model_outer_fold_*.pt")):
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                if 'feature_names' in checkpoint: return checkpoint['feature_names']
            except: pass
        return None

    def _lookup_ablation_dirs(self):
        """Scans for directories containing results for ablated model experiments."""
        base_parent = os.path.abspath(os.path.join(self.output_dir, os.pardir))
        patterns = [os.path.join(base_parent, f"NET*{x}*") for x in ["no_decay", "no_feature", "no_time", "ablate"]]
        found = {}
        for p in patterns:
            for d in glob.glob(p):
                if os.path.isdir(d): found[os.path.basename(d)] = d
        return found

    def _model_mae_rmse_across_folds(self, model_dir):
        """Helper to calculate error stats specifically for ablation study comparisons."""
        files = glob.glob(os.path.join(model_dir, "*predictions_outer_fold_*.csv"))
        rows = []
        for f in files:
            df = pd.read_csv(f)
            rows.append({"fold": df["fold"].iloc[0] if "fold" in df.columns else np.nan, "mae": mean_absolute_error(df["y_true"], df["y_pred"]), "rmse": rmse_compat(df["y_true"], df["y_pred"])})
        return pd.DataFrame(rows).dropna() if rows else None

    def plot_performance_by_race_ethnicity(self, save_path=None):
        """Categorizes accuracy by racial groups to identify potential biases in the model."""
        if self.results_df is None or "ethnicity" not in self.results_df.columns: return
        net_df = prepare_error_data(self.results_df)
        stats_list = []
        for eth in net_df["ethnicity"].dropna().unique():
            subset = net_df[net_df["ethnicity"] == eth]
            stats_list.append({'ethnicity': eth, 'mae': subset["abs_error"].mean(), 'std': subset["abs_error"].std()/np.sqrt(len(subset)), 'n': len(subset)})
        
        stats_df = pd.DataFrame(stats_list).sort_values('mae', ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(stats_df['ethnicity'], stats_df['mae'], yerr=stats_df['std'], capsize=5, alpha=0.75, color='mediumpurple')
        plt.title("Performance by Race/Ethnicity")
        plt.ylabel("MAE")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def table_4_12_ablations(self, save_csv=None):
        """Summarizes performance drops when specific model modules are disabled."""
        base = self._model_mae_rmse_across_folds(self.output_dir)
        if base is None: return None
        ab_dirs = self._lookup_ablation_dirs()
        rows = []
        for name, d in ab_dirs.items():
            ab = self._model_mae_rmse_across_folds(d)
            if ab is None: continue
            m = pd.merge(base, ab, on="fold", suffixes=("_base", "_abl"))
            rows.append({"ablation": name, "delta_MAE": (m["mae_abl"] - m["mae_base"]).mean(), "delta_RMSE": (m["rmse_abl"] - m["rmse_base"]).mean()})
        if rows:
            tab = pd.DataFrame(rows).sort_values("delta_MAE", ascending=False)
            if save_csv: tab.to_csv(save_csv, index=False)
            return tab

    def fig_4_11_tornado_abs_delta_mae(self, ablation_table, save_path=None):
        """Visualizes ablation impact using a tornado plot style."""
        if ablation_table is None: return
        df = ablation_table.copy()
        df["abs_delta"] = df["delta_MAE"].abs()
        plt.figure(figsize=(8, 5))
        plt.barh(df.sort_values("abs_delta")["ablation"], df.sort_values("abs_delta")["abs_delta"])
        plt.xlabel("|ΔMAE|")
        plt.title("Impact of Removed Components")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def fig_4_12_residuals_vs_pred(self, save_path=None, by_model="NET"):
        """Residual plot used to check for heteroscedasticity or non-linear error patterns."""
        if by_model not in self._all_model_results: return
        d = self._all_model_results[by_model].copy()
        d["resid"] = d["y_true"] - d["y_pred"]
        plt.figure(figsize=(8, 6))
        plt.scatter(d["y_pred"], d["resid"], s=12, alpha=0.35)
        plt.axhline(0, color="k", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_kde_distribution(self, save_path=None):
        """Comparison of actual versus predicted INR distributions using Kernel Density Estimation."""
        if self.results_df is None: return
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.results_df['y_true'], label='Actual', alpha=0.7, linewidth=2)
        sns.kdeplot(data=self.results_df['y_pred'], label='Predicted', alpha=0.7, linewidth=2)
        plt.legend()
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_time_attention_heatmap(self, max_patients=10, save_path=None):
        """Generates a heatmap of temporal attention weights for sample patient histories."""
        if not self.attention_data: return
        all_time_attn = []
        for fold_id, data in self.attention_data.items():
            if data['time_attn'] is not None:
                time_attn = data['time_attn']
                if time_attn.ndim == 4: time_attn = time_attn.mean(axis=1)
                if time_attn.ndim == 3: time_attn_diag = [np.diag(attn) for attn in time_attn]
                elif time_attn.ndim == 2: time_attn_diag = [attn for attn in time_attn]
                else: continue
                all_time_attn.extend([np.asarray(attn).flatten() for attn in time_attn_diag if np.asarray(attn).size > 0])

        if not all_time_attn: return
        sel = np.random.choice(len(all_time_attn), min(max_patients, len(all_time_attn)), replace=False)
        max_len = max(len(all_time_attn[i]) for i in sel)
        hm = np.zeros((len(sel), max_len))
        for i, idx in enumerate(sel): hm[i, :len(all_time_attn[idx])] = all_time_attn[idx]

        plt.figure(figsize=(12, 8))
        sns.heatmap(hm, cmap='YlOrRd')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_grouped_mae(self, save_dir):
        """Triggers bar charts for MAE grouped by clinical categories."""
        if self.results_df is None: return
        df_err = prepare_error_data(self.results_df)
        if "sex" in df_err.columns: barplot_mae(df_err, "sex", "MAE by Gender", os.path.join(save_dir, "mae_gender.png"))
        if "age" in df_err.columns: barplot_mae(df_err, "age_group", "MAE by Age", os.path.join(save_dir, "mae_age.png"))
        if "dose" in df_err.columns: barplot_mae(df_err, "dose_group", "MAE by Dose", os.path.join(save_dir, "mae_dose.png"))

    def plot_average_time_attention(self, save_path=None, flip='auto'):
        """Calculates and plots the mean attention weight across all patient time steps."""
        if not self.attention_data: return

        def _normalize_to_sequences(attn):
            seqs = []
            if attn.ndim == 1: seqs.append(attn)
            elif attn.ndim == 2: [seqs.append(row) for row in attn]
            elif attn.ndim == 3:
                if attn.shape[1] == attn.shape[2]: [seqs.append(np.diag(M)) for M in attn]
                else: [seqs.append(row) for row in attn[:, -1, :]]
            elif attn.ndim == 4: return _normalize_to_sequences(attn.mean(axis=1))
            return seqs

        seqs = []
        for fold_id, data in self.attention_data.items():
            if data.get('time_attn') is not None:
                seqs.extend(_normalize_to_sequences(np.asarray(data['time_attn'], dtype=float)))

        if not seqs: return
        head = np.mean([np.mean(v[:3]) if len(v) >= 3 else v.mean() for v in seqs])
        tail = np.mean([np.mean(v[-3:]) if len(v) >= 3 else v.mean() for v in seqs])
        if flip == 'auto' and head > tail * 1.25: seqs = [v[::-1] for v in seqs]

        max_len = max(len(s) for s in seqs)
        padded = np.full((len(seqs), max_len), np.nan)
        for i, s in enumerate(seqs): padded[i, -len(s):] = s
        mean_w, std_w = np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)

        plt.figure(figsize=(8, 6))
        xlbl = [f"t-{k}" for k in range(max_len-1, -1, -1)]
        plt.plot(xlbl, mean_w, 'o-', color='mediumslateblue')
        plt.fill_between(xlbl, mean_w - std_w, mean_w + std_w, alpha=0.25, color='darkviolet')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_feature_attention_weights(self, save_path=None):
        """Plots the horizontal bar chart showing which clinical features the model values most."""
        tab = self.table_4_11_feature_attention_summary()
        if tab is not None:
            plt.figure(figsize=(10, 8))
            plt.barh(tab.index, tab['Mean_Weight'], xerr=tab['Fold_Stability_Std'], color='orchid')
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_performance_by_gender(self, save_path=None):
        """Checks for performance differences between male and female patients."""
        if self.results_df is None: return
        gender_col = next((c for c in ["gender", "sex", "gender_original"] if c in self.results_df.columns), None)
        if gender_col:
            stats_df = self.results_df.copy()
            stats_df["abs_error"] = np.abs(stats_df["y_true"] - stats_df["y_pred"])
            res = stats_df.groupby(gender_col)["abs_error"].agg(['mean', 'std', 'count']).reset_index()
            plt.bar(res[gender_col], res['mean'], yerr=res['std']/np.sqrt(res['count']), color='lavender')
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_performance_by_dose_ranges(self, save_path=None, use_dose_category=False):
        """Charts MAE against different doses to identify potential error trends at high/low ranges."""
        if self.results_df is None: return
        df = prepare_error_data(self.results_df)
        col = "dose_category" if use_dose_category and "dose_category" in df.columns else "dose_group"
        if col in df.columns:
            res = df.groupby(col)["abs_error"].agg(['mean', 'std', 'count']).reset_index()
            plt.bar(res[col], res['mean'], yerr=res['std']/np.sqrt(res['count']), color='mediumpurple')
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=300)
            plt.close()

    def create_summary_table(self):
        """Legacy wrapper to save the feature attention summary to a CSV file."""
        return self.table_4_11_feature_attention_summary(save_csv=os.path.join(self.output_dir, "Table_4_11_feature_attention_summary.csv"))

    def create_ablation_analysis(self):
        """Informational note about automated result collection."""
        print("Results collected from existing sibling directories.")

    def generate_all_plots(self, save_dir=None):
        """Comprehensive orchestrator to generate legacy figures and summaries."""
        if not save_dir: save_dir = self.save_dir
        self.plot_kde_distribution(os.path.join(save_dir, "fig2a_kde_distribution.png"))
        self.plot_time_attention_heatmap(save_path=os.path.join(save_dir, "Fig_4_9_time_attention_heatmap.png"))
        self.plot_feature_attention_weights(os.path.join(save_dir, "fig2c_feature_attention.png"))
        self.plot_average_time_attention(os.path.join(save_dir, "fig2d_avgtime_attention.png"))
        self.plot_performance_by_gender(os.path.join(save_dir, "fig3a_performance_gender.png"))
        self.plot_performance_by_dose_ranges(os.path.join(save_dir, "fig3b_performance_dose.png"))
        self.plot_performance_by_race_ethnicity(os.path.join(save_dir, "fig3c_performance_race_ethnicity.png"))
        self.create_summary_table()
        self.plot_grouped_mae(save_dir)

    def generate_paper_figs(self):
        """Generates specific figures and tables required for a formal dissertation document."""
        self.fig_4_4_foldwise_violin_box(os.path.join(self.save_dir, "Fig_4_4_foldwise_errors.png"))
        self.table_4_8_calibration_metrics(os.path.join(self.save_dir, "Table_4_8_calibration_metrics.csv"))
        calres = self.build_calibration_results_equalwidth(n_bins=10)
        self.plot_calibration_curves_per_fold(calres, os.path.join(self.save_dir, "calibration_figs"))
        self.fig_4_5_calibration_curves(top_k=3, save_path=os.path.join(self.save_dir, "Fig_4_5_calibration_curves.png"))
        self.fig_4_6_delta_mae_bar(os.path.join(self.save_dir, "Fig_4_6_delta_MAE_bar.png"))
        self.table_4_11_feature_attention_summary(os.path.join(self.save_dir, "Table_4_11_feature_attention_summary.csv"))
        ab = self.table_4_12_ablations(os.path.join(self.save_dir, "Table_4_12_ablations.csv"))
        self.fig_4_11_tornado_abs_delta_mae(ab, os.path.join(self.save_dir, "Fig_4_11_tornado_abs_delta_MAE.png"))
        self.fig_4_12_residuals_vs_pred(os.path.join(self.save_dir, "Fig_4_12_residuals_vs_pred.png"), self.baseline_name)

def main():
    MODEL_DIRS = {"WarfarinLSTM": "real_NET_outputs_nested_cv"}
    BASELINE = "NET"
    analyzer = WarfarinNETAnalyzer(output_dir=MODEL_DIRS["WarfarinLSTM"], model_dirs=MODEL_DIRS, baseline_name=BASELINE)
    analyzer.generate_complete_comparison_analysis(xgb_predictions_path="figures_xgb/xgboost_predictions.csv", xgb_csv_dir="figures_xgb_cvsubgroups", save_dir="real_complete_analysis_plots")

if __name__ == "__main__":
    main()