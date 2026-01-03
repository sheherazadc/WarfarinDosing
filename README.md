# WarfarinDosing
This dissertation project implements a specialised deep learning framework for predicting clinical medication dosages. It addresses the challenge of irregular clinical visit intervals and multi-variable patient data through a dual-attention LSTM architecture.

## The Project Pipeline
The system follows a rigorous clinical machine learning pipeline:

1.  **Data Ingestion**: Loading raw medical records (e.g., MIMIC-IV) (as shown in Pipeline.pdf)
2.  **Preprocessing**: Handling missing values, mapping ethnicities, and grouping data into treatment episodes.
3.  **Temporal Windowing**: Converting flat patient history into sequential windows for time-series modeling.
4.  **Nested Cross-Validation**: 
    * **Inner Loop**: Hyperparameter optimization using Optuna.
    * **Outer Loop**: Unbiased performance evaluation on held-out patient groups.
5.  **Interpretability Analysis**: Extracting attention weights to visualize model decision-making.
6.  **Clinical Optimisation**: Using Bayesian Optimization for dose-finding recommendations.
