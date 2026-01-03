import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def filter_treatment_episodes(df, patient_id_col='patient_id', date_col='treatment_date', 
                             max_gap_days=100, inr_col='inr', max_treatment_days=100):
    """
    Cleans up patient timelines. If a patient leaves and comes back after a long time,
    we pick their most active period and ignore the rest to keep the data clean.
    """
    
    # Standardize dates so we can do math on them later
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sorting ensures we see the patient's journey in chronological order
    df_sorted = df.sort_values([patient_id_col, date_col])
    
    patients_to_keep = []
    
    for patient_id, patient_data in df_sorted.groupby(patient_id_col):
        patient_data = patient_data.sort_values(date_col)
        dates = patient_data[date_col].values
        
        # We look for gaps in time that are too long to be part of the same treatment
        time_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
        episode_breaks = np.where(time_diffs > max_gap_days)[0] + 1
        
        # Define where each distinct treatment period starts and ends
        boundaries = np.concatenate([[0], episode_breaks, [len(patient_data)]])
        
        if len(boundaries) == 2:  
            # Only one episode found, so no need to choose
            selected_episode = patient_data
        else:
            # Multiple episodes found. We want the one with the most data points.
            episode_info = []
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                episode_data = patient_data.iloc[start_idx:end_idx]
                
                # We prioritize the episode with the most actual INR readings
                inr_count = episode_data[inr_col].notna().sum()
                episode_info.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'inr_count': inr_count,
                    'data': episode_data
                })
            
            # This picks the winner based on the count we just did
            best_episode = max(episode_info, key=lambda x: x['inr_count'])
            selected_episode = best_episode['data']
            

        # We want the timeline to start at Day 0 for this specific episode
        selected_episode = selected_episode.copy()
        first_time = selected_episode[date_col].min()
        selected_episode['treatment_day'] = (selected_episode[date_col] - first_time).dt.total_seconds() / (24 * 3600)
        
        # Cut off the data if it goes past our maximum window
        selected_episode = selected_episode[selected_episode['treatment_day'] <= max_treatment_days]
        
        if len(selected_episode) > 0:
            patients_to_keep.append(selected_episode)
    
    # Glue all the selected patient slices back together
    result_df = pd.concat(patients_to_keep, ignore_index=True)
    
    print(f"Original data: {len(df)} rows from {df[patient_id_col].nunique()} patients")
    print(f"Filtered data: {len(result_df)} rows from {result_df[patient_id_col].nunique()} patients")
    
    return result_df

def preprocess_data(df: pd.DataFrame, preserve_demographics: bool = True) -> pd.DataFrame:
    """
    Main cleaning hub. This handles column merging, mapping, and outliers.
    """
    df = df.copy()

    # Get a baseline for when the treatment started
    df['inr_datetime'] = pd.to_datetime(df['inr_time'])
    first_visit = df.groupby('subject_id')['inr_datetime'].transform('min')
    df['treatment_day'] = (df['inr_datetime'] - first_visit).dt.total_seconds() / (24 * 3600)
    
    print("Before episode filtering:")
    print(df.groupby('subject_id').head(5)[['subject_id','treatment_day', 'inr_time']].head(20).to_string(index=False))

    # Combine specific conditions into broader clinical categories
    df['cardiac_history'] = df[['afib', 'chf', 'prosthetic_valve', 'arterial_embolism']].max(axis=1)
    df['vascular_disease'] = df[['vte', 'prior_stroke_tia', 'arterial_embolism', 'hypertension']].max(axis=1)
    df['chronic_kidney_disease'] = df['chronic_kidney_dz']
    
    # Drop individual components now that we have the combined flags
    df = df.drop(columns=[
        'afib','chf','prosthetic_valve','arterial_embolism',
        'vte','prior_stroke_tia','chronic_kidney_dz','hypertension'
    ], errors='ignore')

    # Strip out features that aren't useful for this specific model
    df = df.drop(columns=[
        'major_surgery','smoking_status','albumin','creatinine_clearance',
        'pregnancy_status','hematocrit','ast','bun', 'treatment_days'
    ], errors='ignore')

    # Remove rows that are missing critical history
    df = df.dropna(subset=['cancer'])
    df = df.dropna(subset=['previous_inr'])

    # Extra cleanup for columns we don't need
    df = df.drop(columns=['on_pgp_inhibitor','on_pgp_inducer','sodium','potassium'], errors='ignore')

    # Group complex race labels into broader categories for better statistical power
    race_mapping = {
        'WHITE': 'White','WHITE - BRAZILIAN': 'Hispanic/Latino','WHITE - OTHER EUROPEAN': 'White',
        'WHITE - RUSSIAN': 'White','WHITE - EASTERN EUROPEAN': 'White','PORTUGUESE': 'White',
        'BLACK/AFRICAN AMERICAN': 'Black/African American','BLACK/CARIBBEAN ISLAND': 'Black/African American',
        'BLACK/CAPE VERDEAN': 'Black/African American','BLACK/AFRICAN': 'Black/African American',
        'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic/Latino','HISPANIC OR LATINO': 'Hispanic/Latino',
        'HISPANIC/LATINO - DOMINICAN': 'Hispanic/Latino','HISPANIC/LATINO - MEXICAN': 'Hispanic/Latino',
        'HISPANIC/LATINO - SALVADORAN': 'Hispanic/Latino','HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic/Latino',
        'HISPANIC/LATINO - HONDURAN': 'Hispanic/Latino','HISPANIC/LATINO - COLUMBIAN': 'Hispanic/Latino',
        'HISPANIC/LATINO - GUATEMALAN': 'Hispanic/Latino','HISPANIC/LATINO - CUBAN': 'Hispanic/Latino',
        'SOUTH AMERICAN': 'Hispanic/Latino','ASIAN - CHINESE': 'Asian','ASIAN': 'Asian',
        'ASIAN - ASIAN INDIAN': 'Asian','ASIAN - SOUTH EAST ASIAN': 'Asian','ASIAN - KOREAN': 'Asian',
        'AMERICAN INDIAN/ALASKA NATIVE': 'Other','NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Other',
        'MULTIPLE RACE/ETHNICITY':'Other','OTHER': 'Other','UNKNOWN': 'Other','UNABLE TO OBTAIN': 'Other',
        'PATIENT DECLINED TO ANSWER': 'Other'
    }
    df['ethnicity'] = df['race'].map(race_mapping).fillna('Other')
    df.drop(columns=['race'], inplace=True, errors='ignore')

    # Simplify admission types to clarify the urgency of the hospital visit
    adm_map = {
        'EW EMER.': 'Emergent/Urgent','DIRECT EMER.': 'Emergent/Urgent','URGENT': 'Emergent/Urgent',
        'OBSERVATION ADMIT': 'Observation','EU OBSERVATION': 'Observation','DIRECT OBSERVATION': 'Observation',
        'AMBULATORY OBSERVATION': 'Observation','SURGICAL SAME DAY ADMISSION': 'Elective/Same-day',
        'ELECTIVE': 'Elective/Same-day'
    }
    df['admission_type'] = df['admission_type'].map(adm_map)

    # Use clipping to prevent extreme outliers from skewing the model
    df['height_cm'] = df['height_cm'].clip(lower=120, upper=220)
    df['weight_kg'] = df['weight_kg'].clip(lower=40,  upper=200)
    df["ethnicity_asian"] = (df["ethnicity"] == "Asian").astype(int)

    # If we want to do detailed demographic analysis later, we save the labels here
    if preserve_demographics:
        if 'sex' in df.columns:
            df['gender_original'] = df['sex'] 
        
        # Categorize dosage amounts for easier visualization
        if 'dose' in df.columns:
            df['dose_category'] = pd.cut(df['dose'], 
                                       bins=[0, 2.5, 5.0, 7.5, 10.0, float('inf')], 
                                       labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                       include_lowest=True)
            
        # Group ages into buckets
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 30, 50, 65, 80, float('inf')],
                                   labels=['<30', '30-49', '50-64', '65-79', '80+'],
                                   include_lowest=True)
        
        # Standard BMI calculation and grouping
        if 'height_cm' in df.columns and 'weight_kg' in df.columns:
            df['bmi_temp'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
            df['bmi_category'] = pd.cut(df['bmi_temp'], 
                                      bins=[0, 18.5, 25, 30, float('inf')],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
                                      include_lowest=True)
            df = df.drop('bmi_temp', axis=1)


    # Final timestamp cleanup
    df['inr_datetime'] = pd.to_datetime(df['inr_time'], utc=True)
    
    print(df.groupby('subject_id').head(10)[['subject_id','treatment_day'] + 
        ([c for c in ['inr_time'] if c in df.columns])].head(30).to_string(index=False))

    print("columns:", df.columns.tolist())
    return df

def create_demographic_metadata(df, indices):
    """
    Grabs demographic info for specific rows. 
    Useful for checking model fairness across different groups.
    """
    metadata = {}
    
    if len(indices) == 0:
        return metadata
    
    try:
        # Check that we aren't asking for data rows that don't exist
        valid_indices = [i for i in indices if 0 <= i < len(df)]
        if len(valid_indices) != len(indices):
            print(f"Warning: {len(indices) - len(valid_indices)} indices were out of bounds")
        
        if len(valid_indices) == 0:
            return metadata
        
        subset = df.iloc[valid_indices]
        
        # Pull out various labels for the requested indices
        if 'gender_original' in df.columns:
            metadata['gender'] = subset['gender_original'].values
        
        if 'dose' in df.columns:
            metadata['dose'] = subset['dose'].values
            
        if 'dose_category' in df.columns:
            metadata['dose_category'] = subset['dose_category'].astype(str).values
        
        if 'age' in df.columns:
            metadata['age'] = subset['age'].values
            
        if 'age_group' in df.columns:
            metadata['age_group'] = subset['age_group'].astype(str).values
            
        if 'ethnicity_asian' in df.columns:
            metadata['ethnicity_asian'] = subset['ethnicity_asian'].values
            
        if 'ethnicity' in df.columns:
            metadata['ethnicity'] = subset['ethnicity'].values
            
        if 'bmi_category' in df.columns:
            metadata['bmi_category'] = subset['bmi_category'].astype(str).values
            
        if 'subject_id' in df.columns:
            metadata['subject_id'] = subset['subject_id'].values
    
    except Exception as e:
        print(f"Error extracting demographic metadata: {e}")
        return {}
    
    return metadata


def analyze_episode_gaps(df, patient_id_col='patient_id', date_col='treatment_date'):
    """
    Helps us decide the best 'max_gap_days' value by looking at 
    how often patients actually have gaps in their records.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values([patient_id_col, date_col])
    
    all_gaps = []
    patients_with_gaps = 0
    
    for patient_id, patient_data in df_sorted.groupby(patient_id_col):
        patient_data = patient_data.sort_values(date_col)
        dates = patient_data[date_col].values
        
        if len(dates) > 1:
            time_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)
            all_gaps.extend(time_diffs)
            
            max_gap = max(time_diffs)
            if max_gap > 100:  
                patients_with_gaps += 1
    
    gaps_array = np.array(all_gaps)
    
    # Report the distribution of gaps to the console
    print(f"Gap analysis:")
    print(f"- Total measurement gaps: {len(gaps_array)}")
    print(f"- Patients with gaps >100 days: {patients_with_gaps}")
    print(f"- Gap percentiles:")
    print(f"  50th: {np.percentile(gaps_array, 50):.1f} days")
    print(f"  90th: {np.percentile(gaps_array, 90):.1f} days") 
    print(f"  95th: {np.percentile(gaps_array, 95):.1f} days")
    print(f"  99th: {np.percentile(gaps_array, 99):.1f} days")
    print(f"  Max: {np.max(gaps_array)} days")
    
    return gaps_array


NUM_COLS_90 = [
    'previous_inr','alt','dose',
    'weight_kg','bilirubin','height_cm','hemoglobin',
    'platelet','age','creatinine','bmi', 'previous_dose','previous_inr_timediff_hours', 'dose_diff_hours',
]
BIN_COLS_90 = [
    'on_cyp2c9_inhibitor','on_cyp2c9_inducer','on_cyp3a4_inducer',
    'on_cyp3a4_inhibitor','cardiac_history', 'ethnicity_asian'
]

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Prevents crazy outlier values from breaking the model by 
    squishing them down to a specific number of standard deviations.
    """
    def __init__(self, z=2.326, columns=None):
        self.z = z
        self.columns = columns 

    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        use_cols = self.columns or Xdf.columns
        mu = Xdf[use_cols].mean()
        # We replace zero-variance columns with 1 to avoid division errors
        sd = Xdf[use_cols].std().replace(0, 1)
        self.lower_ = (mu - self.z * sd).to_dict()
        self.upper_ = (mu + self.z * sd).to_dict()
        self.columns_ = list(use_cols)
        return self

    def transform(self, X):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        Xdf = Xdf.copy()
        for c in self.columns_:
            Xdf[c] = Xdf[c].clip(self.lower_[c], self.upper_[c])
        return Xdf if isinstance(X, pd.DataFrame) else Xdf.values

class HeightWeightGroupImputer(BaseEstimator, TransformerMixin):
    """
    Smart missing value filler. Instead of a global average, it uses 
    averages based on the patient's age and sex.
    """
    def __init__(self, sex_col='sex', age_col='age',
                 age_bins=(18,30,40,50,60,70,80,np.inf),
                 age_labels=('18-29','30-39','40-49','50-59','60-69','70-79','80+'),
                 cols=('weight_kg','height_cm')):
        self.sex_col = sex_col
        self.age_col = age_col
        self.age_bins = age_bins
        self.age_labels = age_labels
        self.cols = cols

    def fit(self, X, y=None):
        df = X.copy()
        df[self.cols[0]] = pd.to_numeric(df[self.cols[0]], errors='coerce')
        df[self.cols[1]] = pd.to_numeric(df[self.cols[1]], errors='coerce')
        df['__age_band'] = pd.cut(df[self.age_col], bins=self.age_bins,
                                  labels=self.age_labels, right=False)
        # We learn the averages from the training data only
        grp = df.groupby([self.sex_col, '__age_band'], observed=False)[list(self.cols)].mean()
        self.grp_means_ = grp
        return self

    def transform(self, X):
        df = X.copy()
        df[self.cols[0]] = pd.to_numeric(df[self.cols[0]], errors='coerce')
        df[self.cols[1]] = pd.to_numeric(df[self.cols[1]], errors='coerce')
        df['__age_band'] = pd.cut(df[self.age_col], bins=self.age_bins,
                                  labels=self.age_labels, right=False)
        # Fill the NaNs using the averages we calculated in 'fit'
        for (sex, band), row in self.grp_means_.iterrows():
            mask = (df['sex'] == sex) & (df['__age_band'] == band)
            for c in self.cols:
                df.loc[mask & df[c].isna(), c] = row[c]
        df.drop(columns='__age_band', inplace=True)
        return df

def add_saturating_dose_features(df: pd.DataFrame, k: float = 5.0, ec50: float = 5.0, n: float = 2.0) -> pd.DataFrame:
    """
    Adds non-linear versions of the dose column. 
    This helps the model understand that double the dose doesn't always mean double the effect.
    """
    out = df.copy()
    d = out["dose"].clip(lower=0)
    out["mm_dose"] = d / (k + d)                          
    out["hill_dose"] = (d**n) / (ec50**n + d**n)          
    out["tanh_dose"] = np.tanh(d / (ec50))                
    return out

def _saturating_dose_feature_names(_, input_features):
    return list(input_features) + ["mm_dose", "hill_dose", "tanh_dose"]

saturating_dose_tf = FunctionTransformer(add_saturating_dose_features, validate=False)


def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Body Mass Index from height and weight."""
    out = df.copy()
    h_m = out['height_cm'] / 100.0
    out['bmi'] = out['weight_kg'] / (h_m ** 2)
    return out

def _bmi_feature_names(_, input_features):
    return list(input_features) + ['bmi']

bmi_tf = FunctionTransformer(add_bmi, validate=False)



def add_dose_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies math transforms to dose to help the model pick up patterns."""
    out = df.copy()
    out["log_dose"] = np.log1p(out["dose"])   
    out["sqrt_dose"] = np.sqrt(out["dose"])
    return out

def _dose_feature_names(_, input_features):
    return list(input_features) + ["log_dose", "sqrt_dose"]
dose_tf = FunctionTransformer(add_dose_features, validate=False)


def build_preprocessor(num_cols, bin_cols) -> Pipeline:
    """
    Assembles the full pipeline. This is the main engine for transforming 
    raw data into something a machine learning model can actually use.
    """
    # These columns usually have a long tail of values and need log transformation
    skewed = [c for c in ['previous_inr','previous_inr_timediff_hours','treatment_days','alt','bilirubin',] if c in num_cols]
    other  = [c for c in num_cols if c not in skewed]

    # Pipeline for skewed numbers: Winsorize -> Mean Impute -> Log Transform -> Scale
    skewed_pipe = Pipeline([('winsor', Winsorizer(z=2.326, columns=skewed)), ('imputer', SimpleImputer(strategy='mean')),
                            ('log', FunctionTransformer(np.log1p, validate=False)), ('scaler', StandardScaler())])
    
    # Pipeline for normal numbers: Winsorize -> Mean Impute -> Scale
    numeric_pipe = Pipeline([('winsor', Winsorizer(z=2.326, columns=other)), ('imputer', SimpleImputer(strategy='mean')),
                             ('scaler', StandardScaler())])
    
    # Simple pipeline for binary/categorical values
    bin_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

    # Combine all individual transformations into one step
    ct = ColumnTransformer([('skewed_numeric', skewed_pipe, skewed), ('other_numeric', numeric_pipe, other),
                             ('binary', bin_pipe, bin_cols)], remainder='drop' )

    # The final workflow: fill physical stats -> add BMI -> apply column transforms
    return Pipeline([('hw_imp', HeightWeightGroupImputer()),
                     ('add_bmi', FunctionTransformer(add_bmi, validate=False)),
                     ('cols', ct)])