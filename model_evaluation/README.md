# Validation and Clinical Application

This directory contains the tools used to stress-test the LSTM models, compare their performance against standard machine learning baselines, and translate their predictions into usable clinical recommendations.

## Robustness and Architecture Testing

### plots_py folder
This folder holds all the code used for figures in the dissertation and are visualisations of all the .py files below. 
### Core Visualizations
* **`plot_overallwarfarin_net.py`**: Generates the primary evaluation for Warfarin NET.
  * **Calibration Curves**: Predicted vs. Observed outcomes.
  * **Fairness Bar Charts**: MAE broken down by Sex, Age, and Race.
  * **Attention Maps: both feature and time attention,
* **`plot_warfarinnet.py`**: Used for initial plots for testing the model and to debug.
* **`plot_time_lstm.py`**: Generates the primary evaluation for TimeLSTM

### How to Run
After training is complete and data exists in the `outputs/` folder:
```bash
python plot_overallwarfarin_net.py
```

### ablationcv.py
This script performs "component surgery" on the model. It allows you to systematically disable parts of the architecture—like the feature attention mechanism, the temporal attention, or the time-decay factors—and re-run the training pipeline. By comparing the results, we can quantify exactly how much each innovation contributes to the final accuracy.
* **Usage**: `python ablationcv.py --ablation no_timeatt`

### xgboost_analysis.py
This is a standalone benchmarking utility. It trains and evaluates an optimised XGBoost regressor on the same patient data folds used for the LSTM. This provides a clear baseline to prove that the LSTM's ability to remember patient history actually adds value over standard tree-based models.

## Statistical Comparison and Significance

### statistical_tests.py
Once both the LSTM and XGBoost models are trained, this script runs the maths to ensure our improvements aren't just luck. It performs:
* **Normality Checks**: Using the Shapiro-Wilk test to see if error differences are normally distributed.
* **Parametric/Non-Parametric Tests**: Automatically choosing between paired t-tests or Wilcoxon signed-rank tests to confirm statistical significance.
* **Effect Size**: Calculating Cohen’s $d$ to show the magnitude of the performance gap.

## Clinical Utility and Interpretability

### predict_dose.py
This is the main bridge between the model and actual clinical practice. It uses **Bayesian Optimization** to reverse-engineer a patient’s history. Instead of simply predicting an outcome, the script sweeps through potential dosages to find the exact milligram amount that is most likely to move the patient into their specific target INR range.

### gather_best_attn.py
During large-scale cross-validation, attention data is saved across many individual trial files. This utility gathers the "best" attention maps from every fold and trial, compressing them into a single master file. This is used for population-level analysis to see which clinical features were globally most important for safe dosing.

## Metrics of Success
The evaluation scripts prioritise three main clinical measures:
1. **MAE (Mean Absolute Error)**: The average raw difference between predicted and required dose.
2. **RMSE (Root Mean Squared Error)**: Measures the impact of large errors, which are the most dangerous in a medical context.
3. **Percentage Within 20%**: The industry standard for warfarin dosing, checking how many predicted levels fall within 20% of the target level.
