"""
XGBoost Analysis with KDE Plotting - Standalone Script
Runs XGBoost model on first CV fold and generates KDE plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import preprocess_data, build_preprocessor, NUM_COLS_90, BIN_COLS_90, filter_treatment_episodes 
from train.dataset import WarfarinDataset
from warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
from train.train import train_model

# We use seaborn styles to make the charts look professional and readable for reports
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_kde_combined(df, title, save_path):
    """
    Creates a Kernel Density Estimate (KDE) plot. 
    This is basically a smoothed version of a histogram that lets us see 
    if the model's 'shape' of predictions matches the 'shape' of the real data.
    """
    plt.figure(figsize=(8, 5))

    # The dashed blue line shows what the actual INR distribution looks like
    sns.kdeplot(df["true_inr"], label="Actual INR", linestyle="--", color="blue")

    # The solid orange line shows what our model is predicting
    sns.kdeplot(df["pred_inr"], label="Predicted INR", color="orangered")

    plt.title(title)
    plt.xlabel("INR")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined KDE plot: {save_path}")

def prepare_error_data(X_te, y_true, y_pred):
    """
    Merges the test features with the results so we can slice and dice the 
    errors by demographic groups like age or dose size.
    """
    df = X_te.copy()
    df["true_inr"] = y_true
    df["pred_inr"] = y_pred
    df["abs_error"] = np.abs(df["true_inr"] - df["pred_inr"])
    
    # We bin the age into categories to see if the model struggles with elderly vs younger patients
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 30, 50, 65, 80, 120],
            labels=["<30", "30–50", "50–65", "65–80", "80+"]
        )
    
    # Similarly, we group the doses to check if high-dose or low-dose patients are harder to predict
    if "dose" in df.columns:
        df["dose_group"] = pd.cut(
            df["dose"], bins=[0, 2, 4, 6, 8, 15],
            labels=["0–2", "2–4", "4–6", "6–8", "8+"]
        )
    
    return df

def xgb_mae_analysis(X_te, y_true, y_pred, output_dir="figures_xgb", model_name="XGBoost"):
    """
    Runs the full evaluation suite: calculates metrics, prints a summary, 
    and triggers the plot generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Attach predictions to the feature dataframe
    df = prepare_error_data(X_te, y_true, y_pred)
    
    # Draw the density plot
    plot_kde_combined(
        df, 
        f"{model_name} KDE of INR (Actual vs. Predicted)", 
        os.path.join(output_dir, f"{model_name}_kde_combined.png")
    )
    
    # Standard numerical metrics for model comparison
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    
    print(f"\n{model_name} Performance Summary:")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   Mean True INR: {np.mean(y_true):.4f}")
    print(f"   Mean Predicted INR: {np.mean(y_pred):.4f}")
    print(f"   Test set size: {len(y_true)}")
    
    return df

def main():
    """
    Main workflow: Load data -> Preprocess -> Split -> Train -> Evaluate.
    """
    NUM_COLS = [
        'previous_inr', 'previous_inr_timediff_hours', 'treatment_day', 'alt', 'dose',
        'weight_kg', 'bilirubin', 'height_cm', 'dose_diff_hours', 'hemoglobin',
        'platelet', 'age', 'creatinine', 'previous_dose', 'bmi'
    ]
    BIN_COLS = [
        'on_cyp2c9_inhibitor', 'on_cyp2c9_inducer', 'on_cyp3a4_inducer',
        'on_cyp3a4_inhibitor', 'cardiac_history', 'ethnicity_asian'
    ]

    # Target is INR, and we use a fixed seed for reproducible results
    TARGET = 'inr'
    RANDOM_STATE = 42
    N_SPLITS = 5
    
    # Optimized hyperparameters specifically for this dataset
    best_params = {
        'model__n_estimators': 1664, 
        'model__max_depth': 5, 
        'model__learning_rate': 0.006806103717231326, 
        'model__min_child_weight': 14, 
        'model__subsample': 0.991294590009257, 
        'model__colsample_bytree': 0.6943873017366492
    }
    
    print("Starting XGBoost Analysis...")
    
    # Load data from the CSV
    print("Loading and preprocessing data...")
    df_raw = pd.read_csv("warfarin.csv")
    
    # Standard cleaning followed by grouping measurements into treatment episodes
    df = preprocess_data(df_raw) 
    df = filter_treatment_episodes(
        df, 
        patient_id_col='subject_id', 
        date_col='inr_datetime',  
        max_gap_days=100,
        inr_col='inr'
    )
    
    groups = df['subject_id']
    y = df[TARGET]
    X = df.drop(columns=[TARGET, 'subject_id', 'inr_time'], errors='ignore')
    
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Unique subjects: {groups.nunique()}")
    
    # Stratified Cross-Validation
    # We use StratifiedGroupKFold so that data from the same patient stays in the same fold
    y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    print("\nCreating train/test split (first CV fold)...")
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y_bins, groups)):
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
        print(f"   Train size: {len(X_train)}")
        print(f"   Test size: {len(X_test)}")
        break  # We are only analyzing the first fold for this standalone script
    
    # Feature Scaling and Imputation
    print("\nApplying preprocessing...")
    preprocessor = build_preprocessor(NUM_COLS, BIN_COLS)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"   Processed feature dimensions: {X_train_proc.shape[1]}")
    
    # Model Fitting
    print("\nTraining XGBoost model...")
    clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}
    model = XGBRegressor(**clean_params)
    model.fit(X_train_proc, y_train)
    
    # Prediction
    print("\nMaking predictions...")
    y_pred = model.predict(X_test_proc)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"Predictions completed!")
    print(f"Performance: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    # Post-hoc Analysis
    print("\nRunning MAE analysis and generating plots...")
    df_results = xgb_mae_analysis(
        X_te=X_test,
        y_true=y_test,
        y_pred=y_pred,
        output_dir="figures_xgb",
        model_name="XGBoost"
    )
    
    # Export results to CSV
    print("\nSaving results...")
    results_path = "figures_xgb/xgboost_predictions.csv"
    df_results[['true_inr', 'pred_inr', 'abs_error']].to_csv(results_path, index=False)
    print(f"Saved predictions to: {results_path}")
    
    print(f"\nAnalysis complete! Check the 'figures_xgb/' directory for outputs.")
    
    return df_results

if __name__ == "__main__":
    main()