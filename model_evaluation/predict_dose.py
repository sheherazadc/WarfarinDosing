import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import wandb

from preprocess import preprocess_data 
from time_aware_lstm import TimeLSTM
TARGET = 'inr'

# Use the GPU if available to speed up the optimization loops
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "warfarin.csv"
OUTPUT_DIR = "outputs_nested_cv"
os.environ["WANDB_MODE"] = os.getenv("WANDB_MODE", "offline")

# These columns track the passage of time which is critical for the TimeLSTM memory
TIME_COLS = ["previous_inr_timediff_hours", "dose_diff_hours", "treatment_days"]

def pct_within_20pct(y_true, y_pred):
    """
    Standard clinical accuracy check to see how many predictions 
    stay within a safe 20 percent error margin.
    """
    denom = np.maximum(np.abs(y_true), 1e-8)
    pct_err = np.abs((y_pred - y_true) / denom)
    return float(np.mean(pct_err <= 0.20))

def rmse(y_true, y_pred):
    """Calculates the root mean squared error for the dose predictions."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _load_best_fold_checkpoint():
    """
    Automatically scans the training results to find which version 
    of the model performed best during cross validation.
    """
    summary_csv = os.path.join(OUTPUT_DIR, "nested_cv_summary_metrics.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(
            f"Could not find {summary_csv}. Train with the nested CV script first."
        )

    summ = pd.read_csv(summary_csv)
    if summ.empty or "rmse" not in summ.columns or "fold" not in summ.columns:
        raise ValueError(
            f"{summary_csv} is missing required columns ('fold', 'rmse') or is empty."
        )
    
    # Identify the fold with the lowest error
    best_fold = int(summ.loc[summ["rmse"].idxmin(), "fold"])
    model_path = os.path.join(OUTPUT_DIR, f"best_model_outer_fold_{best_fold}.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best model checkpoint not found: {model_path}."
        )

    ckpt = torch.load(model_path, map_location=DEVICE)
    if "config" not in ckpt:
        raise KeyError("Checkpoint missing 'config'. Re-train with the updated script.")
    config = ckpt["config"]

    # We must use the exact same preprocessor instance that was fitted during training
    preproc_path = config.get("preprocessor_path", None)
    if not preproc_path or not os.path.exists(preproc_path):
        raise FileNotFoundError(
            "Checkpoint config lacks a valid 'preprocessor_path'."
        )
    preprocessor = joblib.load(preproc_path)

    # Rebuild the model structure using the saved hyperparameters
    model = TimeLSTM(
        input_dim=int(config["input_dim"]),
        time_dim=int(config["time_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        attention=bool(config["attention"]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(f"Loaded best fold: {best_fold}")
    return model, preprocessor, config, model_path

def _to_float32_array(X):
    """Helper to ensure data is in the 32-bit float format required by PyTorch."""
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    return np.asarray(X, dtype=np.float32)

def run_optimal_dose_finder(
    row_df: pd.DataFrame,
    model: TimeLSTM,
    preprocessor,
    feature_columns,
    dose_range=(0.5, 10.0),
    target_range=(2.0, 3.5),
    num_points=50,
    log_plot_path=None,
):
    """
    Finds the dose that gets the patient closest to their target INR.
    It creates a visualization of the dose-response curve for the clinician.
    """
    assert row_df.shape[0] == 1, "Input should be a single-row DataFrame"

    # We first sweep through a range of doses to visualize the model behavior
    dose_values = np.linspace(dose_range[0], dose_range[1], num_points)
    predictions = []

    for dose in dose_values:
        row_copy = row_df.copy()
        row_copy.loc[:, "dose"] = float(dose)
        row_prep = preprocessor.transform(row_copy.reindex(columns=feature_columns))
        row_prep = _to_float32_array(row_prep)
        time_vals = row_copy[TIME_COLS].values.reshape(1, -1).astype(np.float32)

        with torch.no_grad():
            pred = model(
                torch.from_numpy(row_prep).unsqueeze(1).to(DEVICE),
                torch.from_numpy(time_vals).unsqueeze(1).to(DEVICE),
            ).cpu().numpy()
        predictions.append(float(pred.item()))

    # Build the dose-response visualization
    plt.figure(figsize=(8, 6))
    plt.plot(dose_values, predictions, label="Predicted INR", marker="o")
    plt.xlabel("Dose (mg)")
    plt.ylabel("Predicted INR")
    plt.title("Dose vs Predicted INR")
    plt.grid(True)
    plt.axhspan(*target_range, alpha=0.3, label="Target INR Range")
    midpoint = float(np.mean(target_range))

    # We use Bayesian Optimization to search the dose space intelligently
    def objective(dose):
        row_mod = row_df.copy()
        row_mod.loc[:, "dose"] = float(dose)
        row_prep = preprocessor.transform(row_mod.reindex(columns=feature_columns))
        row_prep = _to_float32_array(row_prep)
        time_vals = row_mod[TIME_COLS].values.reshape(1, -1).astype(np.float32)
        with torch.no_grad():
            pred = model(
                torch.from_numpy(row_prep).unsqueeze(1).to(DEVICE),
                torch.from_numpy(time_vals).unsqueeze(1).to(DEVICE),
            ).cpu().numpy()
        # The objective is to minimize the squared distance to the center of the target range
        return -((float(pred.item()) - midpoint) ** 2)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"dose": (float(dose_range[0]), float(dose_range[1]))},
        random_state=42,
        verbose=0,
    )
    optimizer.maximize(init_points=5, n_iter=25)
    optimal_dose = float(optimizer.max["params"]["dose"])

    # Final validation of the selected dose
    row_opt = row_df.copy()
    row_opt.loc[:, "dose"] = optimal_dose
    row_prep = preprocessor.transform(row_opt.reindex(columns=feature_columns))
    row_prep = _to_float32_array(row_prep)
    time_vals = row_opt[TIME_COLS].values.reshape(1, -1).astype(np.float32)
    with torch.no_grad():
        optimal_pred = model(
            torch.from_numpy(row_prep).unsqueeze(1).to(DEVICE),
            torch.from_numpy(time_vals).unsqueeze(1).to(DEVICE),
        ).cpu().numpy()
    optimal_inr = float(optimal_pred.item())

    # Add the recommended point to the plot
    plt.scatter([optimal_dose], [optimal_inr], label="Optimal Dose", s=100, zorder=5)
    plt.annotate(
        f"Dose: {optimal_dose:.2f}\nINR: {optimal_inr:.2f}",
        xy=(optimal_dose, optimal_inr),
        xytext=(optimal_dose, optimal_inr + 0.1),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
        ha="center",
    )
    plt.legend()

    if log_plot_path is not None:
        os.makedirs(os.path.dirname(log_plot_path), exist_ok=True)
        plt.savefig(log_plot_path, bbox_inches="tight", dpi=300)
        if wandb.run is not None:
            wandb.log({"dose_vs_inr_plot": wandb.Image(log_plot_path)})
    plt.close()

    subject_id = row_df.get("subject_id", pd.Series(["unknown"])).values[0]
    return optimal_dose, optimal_inr, subject_id

if __name__ == "__main__":
    # Load the raw data and prepare it for transformation
    df_raw = pd.read_csv(DATA_PATH)
    df = preprocess_data(df_raw)

    X_df = df.drop(columns=["subject_id", TARGET])
    FEATURE_COLUMNS = X_df.columns
    X_time = df[TIME_COLS].values.astype(np.float32)

    # Initialize the trained model and the matching preprocessor
    model, preprocessor, config, model_path = _load_best_fold_checkpoint()

    # Apply the same transformations used during training to the new data
    X = preprocessor.transform(X_df.reindex(columns=FEATURE_COLUMNS))
    X = X.astype(np.float32)

    # Dimensionality checks to prevent tensor shape errors
    assert X.shape[1] == int(config["input_dim"]), "Feature dim mismatch."
    assert X_time.shape[1] == int(config["time_dim"]), "Time dim mismatch."

    # Process a small sample of random patients to generate recommendations
    os.makedirs("dose_plots", exist_ok=True)
    os.makedirs("dose_logs", exist_ok=True)
    csv_path = "dose_logs/dose_predictions.csv"
    results = []

    np.random.seed(300)
    n_pick = min(5, df["subject_id"].nunique())
    unique_subjects = np.random.choice(df["subject_id"].unique(), size=n_pick, replace=False)

    for sid in unique_subjects:
        # Use the most recent visit data for the recommendation
        row_df = df[df["subject_id"] == sid].iloc[[0]]

        plot_path = f"dose_plots/dose_vs_inr_subject_{sid}.png"
        optimal_dose, optimal_inr, subject_id = run_optimal_dose_finder(
            row_df=row_df,
            model=model,
            preprocessor=preprocessor,
            feature_columns=FEATURE_COLUMNS,
            log_plot_path=plot_path
        )
        results.append(
            {"subject_id": subject_id, "optimal_dose": round(optimal_dose, 2), "predicted_inr": round(optimal_inr, 2)}
        )

    # Export all recommendations to a central file
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"All dose predictions saved to: {csv_path}")