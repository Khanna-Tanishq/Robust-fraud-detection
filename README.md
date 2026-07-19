```markdown
# 🕵️‍♂️ Robust Fraud Detection Pipeline 📊

Detecting financial fraud is an exercise in finding a needle in a haystack. In this dataset, legitimate transactions account for over 99% of the data. A standard model that simply predicts every transaction as legitimate would achieve near-perfect accuracy—while catching zero actual fraud. 

This project discards the illusion of accuracy in favor of **algorithmic precision**. 

Built as an interactive **Streamlit web application**, this project features a mathematically secure, zero-leakage machine learning pipeline. By utilizing `imblearn` and the Synthetic Minority Over-sampling Technique (SMOTE), the model interpolates synthetic data specifically within the training folds during Cross-Validation. This prevents the model from simply memorizing duplicated data or peeking at the test set.

---

## 📂 Repository Structure

*   **`Model/`**: Contains the exported machine learning pipeline (`.pkl` file) trained with SMOTE and the final classifier (e.g., Random Forest).
*   **`Notebooks/`**: Includes the Google Colab Jupyter Notebooks detailing the data exploration, pipeline construction, GridSearch hyperparameter tuning, and metric evaluation.
*   **`Project Statement/`**: Contains the original project requirements, constraints, and documentation for the Industrial Training Kit.
*   **`app.py`**: The main Python script that runs the interactive Streamlit web application.
*   **`requirements.txt`**: A list of all required Python dependencies to run the project.

---

## 🚀 How to Run the App Locally

If you want to run this application on your own machine, follow these steps:

### 1. Clone the repository
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/Khanna-Tanishq/Robust-fraud-detection.git](https://github.com/Khanna-Tanishq/Robust-fraud-detection.git)
cd Robust-fraud-detection

```

### 2. Install dependencies

It is recommended to use a virtual environment. Install the required packages using:

```bash
pip install -r requirements.txt

```

### 3. Launch the Streamlit app

Run the following command to start the local server:

```bash
streamlit run app.py

```

### 4. Access the app

The app will automatically open in your default web browser. If it doesn't, navigate to `http://localhost:8501` in your browser's address bar. You can now upload a CSV file of transaction data to scan for potential fraud!

---

> **💡 Note on Model Pathing:**
> Since the model is inside the `Model/` folder, ensure that inside your `app.py` file, the path points to that specific folder when loading the `.pkl` file.
> Example: `pipeline = joblib.load('Model/fraud_model.pkl')`

---

*Developed by Tanishq Khanna*

```

```
