import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from warfarinnet_lstm import WarfarinNetLSTM as NETLSTM
from train.dataset import WarfarinDataset
from torch.utils.data import DataLoader

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def prepare_error_data(df):
    """
    Groups errors into bins so we can see if the model is 
    less accurate for specific patient populations.
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
            df["dose"], bins=[0, 2, 4, 6, 8, 15],
            labels=["0–2", "2–4", "4–6", "6–8", "8+"]
        )
    return df


def barplot_mae(df, group_col, title, save_path):
    """
    Creates a bar chart comparing Mean Absolute Error across groups.
    The error bars show the standard deviation within that group.
    """
    grouped = df.groupby(group_col)['abs_error'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(7, 5))
    plt.bar(grouped[group_col].astype(str),
            grouped['mean'],
            yerr=grouped['std'],
            capsize=5,
            alpha=0.75,
            color='teal')

    plt.title(title)
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.xlabel(group_col.replace("_", " ").title())
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved MAE bar plot: {save_path}")
    return grouped


class WarfarinNETAnalyzer:
    """
    This class acts as a control center for evaluating our model. 
    It pulls in predictions, attention weights, and raw models to create 
    the final report figures.
    """
    
    def __init__(self, output_dir="NET_outputs_nested_cv"):
        self.output_dir = output_dir
        self.results_df = None
        self.attention_data = {}
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # We need the exact order of features to match them with attention weights
        self.feature_names = [
            'previous_inr', 'previous_inr_timediff_hours', 'treatment_days', 'alt', 'dose',
            'weight_kg', 'bilirubin', 'height_cm', 'dose_diff_hours', 'hemoglobin',
            'platelet', 'age', 'creatinine', 'previous_dose', 'bmi', 
            'on_cyp2c9_inhibitor', 'on_cyp2c9_inducer', 'on_cyp3a4_inducer',
            'on_cyp3a4_inhibitor', 'cardiac_history', 'ethnicity_asian'
        ]
        
        self._load_results()
        self._load_attention_data()
        self._load_models()
    
    def _load_results(self):
        """Finds and combines all cross-validation CSVs into one master list."""
        pred_files = glob.glob(os.path.join(self.output_dir, "NET_predictions_outer_fold_*.csv"))
        if pred_files:
            dfs = []
            for file in pred_files:
                df = pd.read_csv(file)
                dfs.append(df)
            self.results_df = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {len(pred_files)} prediction files")
        else:
            print("No prediction files found")
    
    def _load_attention_data(self):
        """Loads the .npz files containing the time and feature importance scores."""
        attn_files = glob.glob(os.path.join(self.output_dir, 'WarfarinNet', "attn_outputs_trial_*_best.npz"))
        for file in attn_files:
            try:
                data = np.load(file, allow_pickle=True)
                fold_id = os.path.basename(file).split('_')[3]  
                self.attention_data[fold_id] = {
                    'preds': data['preds'],
                    'targets': data['targets'],
                    'time_attn': data['attn_time'],
                    'feat_attn': data['attn_feat']
                }
                print(f"Loaded attention data for fold {fold_id}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def _load_models(self):
        """Reconstructs the trained models from saved state dictionaries."""
        model_files = glob.glob(os.path.join(self.output_dir, "NET_best_model_outer_fold_*.pt"))
        for file in model_files:
            try:
                fold_id = os.path.basename(file).split('_')[-1].replace('.pt', '')
                checkpoint = torch.load(file, map_location=self.device)
                config = checkpoint['config']
                
                model = NETLSTM(
                    input_dim=config['input_dim'],
                    time_dim=config['time_dim'],
                    hidden_size=config['hidden_size'],
                    num_layers=config['num_layers'],
                    dropout=config['dropout'],
                    attention=config['attention']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models[fold_id] = {
                    'model': model,
                    'config': config
                }
            except Exception as e:
                print(f"Error loading model {file}: {e}")

    def plot_kde_distribution(self, save_path=None):
        """
        Visualizes the overlap between the real INR values and the model's 
        guesses. A good model will have two curves that look almost identical.
        """
        if self.results_df is None:
            return
            
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=self.results_df['y_true'], label='Actual INR', alpha=0.7, linewidth=2)
        sns.kdeplot(data=self.results_df['y_pred'], label='Predicted INR', alpha=0.7, linewidth=2)
        
        plt.xlabel('INR Value')
        plt.ylabel('Density')
        plt.title('Actual vs Predicted INR Distributions')
        plt.legend()
        
        # We calculate statistics to see if the distributions are mathematically different
        ks_stat, ks_p = stats.ks_2samp(self.results_df['y_true'], self.results_df['y_pred'])
        plt.text(0.02, 0.98, f'KS Test p-value: {ks_p:.4f}',
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_time_attention_heatmap(self, max_patients=10, save_path=None):
        """
        Shows how the model 'looked back' at history for different patients.
        Brighter colors mean the model paid more attention to that specific visit.
        """
        if not self.attention_data:
            return

        all_time_attn = []
        for fold_id, data in self.attention_data.items():
            if data['time_attn'] is not None:
                time_attn = data['time_attn']
                if time_attn.ndim == 4:
                    time_attn = time_attn.mean(axis=1)
                # We grab the diagonal weights which represent self-attention per visit
                if time_attn.ndim == 3:
                    time_attn_diag = [np.diag(attn) for attn in time_attn]
                else:
                    time_attn_diag = [time_attn]
                all_time_attn.extend(time_attn_diag)

        n_patients = min(max_patients, len(all_time_attn))
        selected_patients = np.random.choice(len(all_time_attn), n_patients, replace=False)

        max_seq_len = max(len(all_time_attn[i]) for i in selected_patients)
        heatmap_data = np.zeros((n_patients, max_seq_len))
        for i, idx in enumerate(selected_patients):
            attn = all_time_attn[idx]
            heatmap_data[i, :len(attn)] = attn

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'})
        plt.title('Attention weights across patient visits')
        plt.xlabel('Visit steps (0 = Oldest)')
        plt.ylabel('Randomly selected test patients')

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


    def plot_grouped_mae(self, save_dir):
        """Generates all demographic MAE charts in one go."""
        if self.results_df is None:
            return

        df_err = prepare_error_data(self.results_df)

        if "sex" in df_err.columns:
            barplot_mae(df_err, "sex", "MAE by Gender",
                        os.path.join(save_dir, "mae_gender.png"))

        if "age" in df_err.columns:
            barplot_mae(df_err, "age_group", "MAE by Age",
                        os.path.join(save_dir, "mae_age.png"))

        if "dose" in df_err.columns:
            barplot_mae(df_err, "dose_group", "MAE by Dose",
                        os.path.join(save_dir, "mae_dose.png"))

    
    def plot_average_time_attention(self, save_path=None):
        """
        Aggregates every patient's attention map to see the general trend.
        This usually shows that the model prioritizes the most recent visits.
        """
        if not self.attention_data:
            return

        all_time_attn = []
        for fold_id, data in self.attention_data.items():
            if data['time_attn'] is not None:
                time_attn = data['time_attn']
                if time_attn.ndim == 3:
                    time_attn = time_attn[:, -1, :]
                for seq in time_attn:
                    all_time_attn.append(np.array(seq))

        max_len = max(len(seq) for seq in all_time_attn)
        padded = np.zeros((len(all_time_attn), max_len))
        for i, seq in enumerate(all_time_attn):
            padded[i, :len(seq)] = seq

        mean_weights = np.mean(padded, axis=0)
        std_weights = np.std(padded, axis=0)
        x_labels = [f"t-{max_len-1-i}" for i in range(max_len)]

        plt.figure(figsize=(10, 6))
        plt.plot(x_labels, mean_weights, marker='o', label='Mean Attention')
        plt.fill_between(x_labels, mean_weights - std_weights, mean_weights + std_weights, alpha=0.2)
        plt.title('Recency bias in model attention')
        plt.xlabel('Visits relative to current prediction')
        plt.ylabel('Weight')
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_feature_attention_weights(self, save_path=None):
        """
        Creates a horizontal bar chart of feature importance.
        Features are color-coded to highlight the high-impact variables.
        """
        if not self.attention_data:
            return
            
        all_feat_attn = []
        for fold_id, data in self.attention_data.items():
            if data['feat_attn'] is not None:
                all_feat_attn.append(data['feat_attn'])
        
        combined_feat_attn = np.concatenate(all_feat_attn, axis=0)
        mean_feat_weights = combined_feat_attn.mean(axis=0)
        std_feat_weights = combined_feat_attn.std(axis=0)
        
        n_features = min(len(mean_feat_weights), len(self.feature_names))
        feat_df = pd.DataFrame({
            'Feature': self.feature_names[:n_features],
            'Mean_Weight': mean_feat_weights[:n_features],
            'Std_Weight': std_feat_weights[:n_features]
        }).sort_values('Mean_Weight', ascending=True)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(feat_df)), feat_df['Mean_Weight'], xerr=feat_df['Std_Weight'], alpha=0.7)
        
        # Color features by importance (High = Red, Mid = Orange, Low = Blue)
        threshold = mean_feat_weights.mean() + mean_feat_weights.std()
        for i, bar in enumerate(bars):
            weight = feat_df.iloc[i]['Mean_Weight']
            if weight > threshold:
                bar.set_color('red')  
            elif weight > mean_feat_weights.mean():
                bar.set_color('orange')  
            else:
                bar.set_color('skyblue')  
        
        plt.yticks(range(len(feat_df)), feat_df['Feature'])
        plt.xlabel('Mean Attention Weight')
        plt.title('Clinical Feature Importance')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_performance_by_gender(self, save_path=None):
        """
        A scatter plot split by gender.
        Useful for checking if the model is biased toward one sex.
        """
        if self.results_df is None:
            return

        gender_col = "sex" if "sex" in self.results_df.columns else "gender_original"
        if gender_col not in self.results_df.columns:
            return

        genders = self.results_df[gender_col].dropna().unique()
        plt.figure(figsize=(10, 6))

        for gender in genders:
            gdata = self.results_df[self.results_df[gender_col] == gender]
            mae = mean_absolute_error(gdata["y_true"], gdata["y_pred"])
            plt.scatter(gdata["y_true"], gdata["y_pred"], alpha=0.5, label=f"{gender} (MAE={mae:.2f})")

        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlabel("Actual INR")
        plt.ylabel("Predicted INR")
        plt.title("Performance by Gender")
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_performance_by_dose_ranges(self, save_path=None, bins=[0, 1, 2, 3, 5, 10]):
        """Categorizes prediction accuracy by how much Warfarin the patient is taking."""
        if self.results_df is None or "dose" not in self.results_df.columns:
            return

        self.results_df["dose_range"] = pd.cut(self.results_df["dose"], bins=bins, include_lowest=True)

        plt.figure(figsize=(12, 8))
        for drange in self.results_df["dose_range"].cat.categories:
            subset = self.results_df[self.results_df["dose_range"] == drange]
            if subset.empty: continue
            mae = mean_absolute_error(subset["y_true"], subset["y_pred"])
            plt.scatter(subset["y_true"], subset["y_pred"], alpha=0.6, label=f"{drange} mg (MAE={mae:.2f})")

        plt.plot([0, 10], [0, 10], 'r--')
        plt.xlabel("Actual INR")
        plt.ylabel("Predicted INR")
        plt.title("Performance by Dose Range")
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def create_summary_table(self):
        """Saves a CSV of feature importance so you can use the raw numbers in a paper."""
        if not self.attention_data:
            return
            
        fold_feat_weights = {}
        for fold_id, data in self.attention_data.items():
            if data['feat_attn'] is not None:
                fold_feat_weights[fold_id] = data['feat_attn'].mean(axis=0)
        
        if fold_feat_weights:
            feat_summary = pd.DataFrame(fold_feat_weights).T
            feat_summary.columns = self.feature_names[:feat_summary.shape[1]]
            
            summary_stats = pd.DataFrame({
                'Mean_Weight': feat_summary.mean(),
                'Fold_Stability_Std': feat_summary.std(),
            }).sort_values('Mean_Weight', ascending=False)
            
            summary_path = os.path.join(self.output_dir, "feature_attention_summary.csv")
            summary_stats.to_csv(summary_path)

    def generate_all_plots(self, save_dir=None):
        """Runs the entire analysis pipeline."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        print("Generating comprehensive Warfarin NET analysis...")
        self.plot_kde_distribution(os.path.join(save_dir, "fig2a_kde.png") if save_dir else None)
        self.plot_time_attention_heatmap(save_path=os.path.join(save_dir, "fig2b_time_attn.png") if save_dir else None)
        self.plot_feature_attention_weights(os.path.join(save_dir, "fig2c_feat_attn.png") if save_dir else None)
        self.plot_average_time_attention(os.path.join(save_dir, "fig2d_avg_time_attn.png") if save_dir else None)
        self.plot_performance_by_gender(os.path.join(save_dir, "fig3a_gender.png") if save_dir else None)
        self.plot_performance_by_dose_ranges(os.path.join(save_dir, "fig3b_dose.png") if save_dir else None)
        self.create_summary_table()
        self.plot_grouped_mae(save_dir)


def main():
    analyzer = WarfarinNETAnalyzer("NET_outputs_nested_cv")
    analyzer.generate_all_plots(save_dir="NET_analysis_plots")


if __name__ == "__main__":
    main()