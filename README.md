# Robust Fraud Detection Pipeline 🕵️‍♂️📊

Detecting financial fraud is an exercise in finding a needle in a haystack. In this dataset, legitimate transactions account for 99.83% of the data, meaning a "lazy" model that simply predicts every transaction as legitimate would achieve 99.83% accuracy—while catching zero actual fraud. 

This project discards the illusion of accuracy in favor of **algorithmic precision**. 

Built as an interactive **Streamlit web application**, this project features a mathematically secure, zero-leakage machine learning pipeline. By utilizing `imblearn` and the Synthetic Minority Over-sampling Technique (SMOTE), the model interpolates synthetic data specifically within the training folds during Cross-Validation. This prevents the model from simply memorizing duplicated data or peeking at the test set.

**Key Technical Features:**
*   **Zero-Leakage Architecture:** Strict isolation of scaling and resampling within the Cross-Validation loop using `imblearn.pipeline`.
*   **Imbalance Handling:** Utilizes SMOTE to generate synthetic minority class representations rather than arbitrary duplication.
*   **Strict Evaluation Metrics:** Models are optimized and evaluated strictly on Precision, Recall, and ROC-AUC to minimize both financial loss (False Negatives) and customer friction (False Positives).
*   **Interactive UI:** Deployed as a mobile-responsive Streamlit application for seamless inference.
