# Training Infrastructure

This module handles the data engineering and the execution of the Nested Cross-Validation (CV) framework.

### Core Files
* **`preprocess.py`**: Handles initial hygiene, outlier clipping, and standardized race/ethnicity mapping.
* **`dataset.py`**: Converts patient records into sliding windows. It ensures patient boundaries are strictly respected to prevent data leakage between sequences.
* **`train.py`**: Manages the training loop, utilizing Automatic Mixed Precision (AMP) and early stopping.
* **`nested_cv_lstm_net.py`**: The main entry point. It separates tuning from final testing to eliminate optimistic bias.

### Setup and Execution

**Dependencies**:
Install all necessary Python libraries via the requirements file:
```bash
pip install -r requirements.txt
```
To begin the training process with 40 trials of hyperparameter optimisation:

```bash
python nested_cv_warfarinnet.py --df warfarin.csv --trials 40 --batch_size 256
```
Automated Batching: Use the provided shell script to run nested_cv_warfarinnet folds at the same time to speed up processing time if gpu allows:

```bash
chmod +x run_all_folds_net.sh
./run_all_folds_net.sh
