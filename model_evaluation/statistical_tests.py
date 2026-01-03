"""
Statistical comparison of LSTM vs XGBoost models using paired t-tests and ANOVA.
Assumes you have:
- NET predictions: real_NET_outputs_nested_cv/NET_predictions_outer_fold_*.csv
- XGBoost predictions: XGB_outputs_nested_cv/XGB_predictions_outer_fold_*.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# We are building the XGBoost results table by hand here since we have the fold numbers ready
import pandas as pd

xgb_metrics = pd.DataFrame({
    'fold': [1, 2, 3, 4, 5],
    'mae': [0.3685, 0.3751, 0.3714, 0.3763, 0.3772],
    'rmse': [0.6861, 0.6870, 0.6608, 0.7012, 0.6638],
    'within20': [0.7560, 0.7474, 0.7458, 0.7476, 0.7450]
})

xgb_metrics.to_csv('xgboost_fold_metrics.csv', index=False)
print("Saved xgboost_fold_metrics.csv")
print(xgb_metrics)

class ModelComparison:
    def __init__(self, net_dir="real_NET_outputs_nested_cv", 
                 xgb_summary_csv="xgboost_fold_metrics.csv"):
        self.net_dir = Path(net_dir)
        self.xgb_summary = xgb_summary_csv
        self.results = {}
        
    def load_fold_metrics(self):
        """
        Gathers the performance scores for both models so we can compare 
        them head-to-head across the same cross-validation folds.
        """
        fold_metrics = []
        
        # Pull in the XGBoost stats we just saved
        xgb_df = pd.read_csv(self.xgb_summary)
        xgb_dict = xgb_df.set_index('fold').to_dict('index')
        
        # We loop through each fold to calculate the NET model's error scores
        for fold in range(1, 6):
            net_file = self.net_dir / f"NET_predictions_outer_fold_{fold}.csv"
            if net_file.exists() and fold in xgb_dict:
                df = pd.read_csv(net_file)
                # Calculating average absolute error and the standard root mean square error
                mae_net = np.abs(df['y_true'] - df['y_pred']).mean()
                rmse_net = np.sqrt(((df['y_true'] - df['y_pred'])**2).mean())
                
                fold_metrics.append({
                    'fold': fold,
                    'NET_MAE': mae_net,
                    'NET_RMSE': rmse_net,
                    'XGB_MAE': xgb_dict[fold]['mae'],
                    'XGB_RMSE': xgb_dict[fold]['rmse']
                })
        
        if not fold_metrics:
            raise ValueError("No matching fold predictions found!")
        
        self.fold_df = pd.DataFrame(fold_metrics)
        print("Loaded metrics for folds:", self.fold_df['fold'].tolist())
        return self.fold_df
    
    def paired_t_test(self, metric='MAE'):
        """
        Checks if the difference in model performance is likely 'real' 
        or just due to a lucky split of the data.
        """
        net_col = f'NET_{metric}'
        xgb_col = f'XGB_{metric}'
        
        net_values = self.fold_df[net_col].values
        xgb_values = self.fold_df[xgb_col].values
        
        # This test is paired because both models were tested on the exact same data folds
        t_stat, p_value = stats.ttest_rel(net_values, xgb_values)
        
        # Cohen's d tells us the 'size' of the difference, regardless of p-value
        differences = net_values - xgb_values
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # This range gives us a 95% certainty of where the true difference lies
        n = len(differences)
        se = std_diff / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=mean_diff, scale=se)
        
        results = {
            'metric': metric,
            'NET_mean': np.mean(net_values),
            'NET_std': np.std(net_values),
            'XGB_mean': np.mean(xgb_values),
            'XGB_std': np.std(xgb_values),
            'mean_difference': mean_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'n_folds': n
        }
        
        self.results[f'paired_t_{metric}'] = results
        return results
    
    def wilcoxon_test(self, metric='MAE'):
        """
        A backup test that works even if our error scores don't 
        follow a nice bell-curve distribution.
        """
        net_col = f'NET_{metric}'
        xgb_col = f'XGB_{metric}'
        
        net_values = self.fold_df[net_col].values
        xgb_values = self.fold_df[xgb_col].values
        
        stat, p_value = stats.wilcoxon(net_values, xgb_values)
        
        # Converts the result into a standardized effect size 'r'
        n = len(net_values)
        z = stats.norm.ppf(1 - p_value/2) * np.sign(np.mean(net_values - xgb_values))
        r = z / np.sqrt(n)
        
        results = {
            'metric': metric,
            'statistic': stat,
            'p_value': p_value,
            'effect_size_r': r,
            'n_folds': n
        }
        
        self.results[f'wilcoxon_{metric}'] = results
        return results
    
    def shapiro_wilk_test(self, metric='MAE'):
        """
        Validates if our data is 'normal' enough to trust the 
        standard t-test results.
        """
        net_col = f'NET_{metric}'
        xgb_col = f'XGB_{metric}'
        
        differences = self.fold_df[net_col] - self.fold_df[xgb_col]
        stat, p_value = stats.shapiro(differences)
        
        print(f"\nShapiro-Wilk test for {metric} differences:")
        print(f"  Statistic: {stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Normal distribution: {'Yes' if p_value > 0.05 else 'No'} (α=0.05)")
        
        return {'statistic': stat, 'p_value': p_value}
    
    def repeated_measures_anova(self):
        """
        Used if we ever add a third model to the mix. 
        For just two models, the paired t-test we did earlier is actually better.
        """
        print("\nNote: For comparing only 2 models, paired t-test is more appropriate.")
        print("Repeated measures ANOVA is useful when comparing 3+ models.")
        
        # Reshape data into a 'long' format required by most ANOVA functions
        long_df = pd.melt(
            self.fold_df,
            id_vars=['fold'],
            value_vars=['NET_MAE', 'XGB_MAE'],
            var_name='model',
            value_name='MAE'
        )
        
        # Running the f-test to see if there is any variance between the groups
        from scipy.stats import f_oneway
        net_mae = self.fold_df['NET_MAE'].values
        xgb_mae = self.fold_df['XGB_MAE'].values
        
        f_stat, p_value = f_oneway(net_mae, xgb_mae)
        
        print(f"\nOne-way ANOVA (MAE):")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        
        return long_df
    
    def print_summary(self):
        """Generates a readable report of all our statistical findings."""
        print("\n" + "="*70)
        print("STATISTICAL COMPARISON: WarfarinNET (LSTM) vs XGBoost")
        print("="*70)
        
        print("\n--- FOLD-WISE METRICS ---")
        print(self.fold_df.to_string(index=False))
        
        # Report for Absolute Error
        print("\n" + "="*70)
        print("PAIRED T-TEST: MAE")
        print("="*70)
        mae_results = self.paired_t_test('MAE')
        print(f"NET MAE:     {mae_results['NET_mean']:.4f} ± {mae_results['NET_std']:.4f}")
        print(f"XGB MAE:     {mae_results['XGB_mean']:.4f} ± {mae_results['XGB_std']:.4f}")
        print(f"Difference:  {mae_results['mean_difference']:.4f} (NET - XGB)")
        print(f"95% CI:      [{mae_results['ci_95_lower']:.4f}, {mae_results['ci_95_upper']:.4f}]")
        print(f"t-statistic: {mae_results['t_statistic']:.4f}")
        print(f"p-value:     {mae_results['p_value']:.4f}")
        print(f"Cohen's d:   {mae_results['cohens_d']:.4f}")
        print(f"Significant: {'Yes' if mae_results['p_value'] < 0.05 else 'No'} (α=0.05)")
        
        # Report for Root Mean Square Error
        print("\n" + "="*70)
        print("PAIRED T-TEST: RMSE")
        print("="*70)
        rmse_results = self.paired_t_test('RMSE')
        print(f"NET RMSE:    {rmse_results['NET_mean']:.4f} ± {rmse_results['NET_std']:.4f}")
        print(f"XGB RMSE:    {rmse_results['XGB_mean']:.4f} ± {rmse_results['XGB_std']:.4f}")
        print(f"Difference:  {rmse_results['mean_difference']:.4f} (NET - XGB)")
        print(f"95% CI:      [{rmse_results['ci_95_lower']:.4f}, {rmse_results['ci_95_upper']:.4f}]")
        print(f"t-statistic: {rmse_results['t_statistic']:.4f}")
        print(f"p-value:     {rmse_results['p_value']:.4f}")
        print(f"Cohen's d:   {rmse_results['cohens_d']:.4f}")
        print(f"Significant: {'Yes' if rmse_results['p_value'] < 0.05 else 'No'} (α=0.05)")
        
        # Check if we should even be using t-tests
        print("\n" + "="*70)
        print("NORMALITY TESTS (Shapiro-Wilk)")
        print("="*70)
        self.shapiro_wilk_test('MAE')
        self.shapiro_wilk_test('RMSE')
        
        # Report non-parametric results just to be thorough
        print("\n" + "="*70)
        print("NON-PARAMETRIC TESTS (Wilcoxon)")
        print("="*70)
        wilcoxon_mae = self.wilcoxon_test('MAE')
        print(f"MAE - Wilcoxon p-value: {wilcoxon_mae['p_value']:.4f}")
        print(f"MAE - Effect size (r): {wilcoxon_mae['effect_size_r']:.4f}")
        
        wilcoxon_rmse = self.wilcoxon_test('RMSE')
        print(f"RMSE - Wilcoxon p-value: {wilcoxon_rmse['p_value']:.4f}")
        print(f"RMSE - Effect size (r): {wilcoxon_rmse['effect_size_r']:.4f}")
    
    def plot_comparison(self, save_path='model_comparison_plots.png'):
        """Visualizes the data so we can see the performance gap manually."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Line chart showing how each model did on every fold
        ax1 = axes[0, 0]
        x = self.fold_df['fold']
        ax1.plot(x, self.fold_df['NET_MAE'], 'o-', label='NET (LSTM)', color='orchid', linewidth=2)
        ax1.plot(x, self.fold_df['XGB_MAE'], 's-', label='XGBoost', color='teal', linewidth=2)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('MAE')
        ax1.set_title('MAE by Fold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Same as above but for the RMSE metric
        ax2 = axes[0, 1]
        ax2.plot(x, self.fold_df['NET_RMSE'], 'o-', label='NET (LSTM)', color='orchid', linewidth=2)
        ax2.plot(x, self.fold_df['XGB_RMSE'], 's-', label='XGBoost', color='teal', linewidth=2)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('RMSE')
        ax2.set_title('RMSE by Fold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bar chart showing exactly which model won each fold and by how much
        ax3 = axes[1, 0]
        mae_diff = self.fold_df['NET_MAE'] - self.fold_df['XGB_MAE']
        ax3.bar(x, mae_diff, color=['red' if d > 0 else 'green' for d in mae_diff], alpha=0.7)
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('MAE Difference (NET - XGB)')
        ax3.set_title('Paired Differences: MAE')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot to show the overall range of performance for both models
        ax4 = axes[1, 1]
        data_to_plot = [self.fold_df['NET_MAE'], self.fold_df['XGB_MAE']]
        bp = ax4.boxplot(data_to_plot, labels=['NET (LSTM)', 'XGBoost'], patch_artist=True)
        bp['boxes'][0].set_facecolor('orchid')
        bp['boxes'][1].set_facecolor('teal')
        ax4.set_ylabel('MAE')
        ax4.set_title('MAE Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plots: {save_path}")
        plt.close()
    
    def save_results(self, output_path='statistical_test_results.csv'):
        """Dumps all the math results into a CSV for future reference."""
        results_list = []
        for test_name, result in self.results.items():
            row = {'test': test_name}
            row.update(result)
            results_list.append(row)
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_path, index=False)
        print(f"Saved results: {output_path}")
        
        return results_df

# Kick off the whole comparison process
if __name__ == "__main__":
    comparison = ModelComparison(
        net_dir="real_NET_outputs_nested_cv",
        xgb_summary_csv="xgboost_fold_metrics.csv"  
    )
    
    # Try to load the files and fail gracefully if they aren't there
    try:
        comparison.load_fold_metrics()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both NET and XGBoost prediction files exist in the specified directories")
        exit(1)
    
    # Run the tests, draw the plots, and save the CSVs
    comparison.print_summary()
    comparison.plot_comparison('model_comparison_statistical.png')
    comparison.save_results('statistical_comparison_results.csv')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("Generated files:")
    print("  - statistical_comparison_results.csv")
    print("  - model_comparison_statistical.png")