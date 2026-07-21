import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Fraud Detection AI", page_icon="🕵️‍♂️", layout="centered")

# 2. Load the Zero-Leakage Pipeline
@st.cache_resource # This caches the model so it doesn't reload on every button click
def load_model():
    return joblib.load('Model/fraud_model.pkl')
pipeline = load_model()

# 3. Build the UI
st.title("Robust Fraud Detection System 🕵️‍♂️")
st.markdown("""
Upload a CSV file containing transaction data. The algorithmic engine will scan the 
transactions and flag potential anomalies using our SMOTE-trained ensemble model.
""")

# 4. File Uploader
uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
# 5. Prediction Engine
    if st.button("Scan for Fraud"):
        with st.spinner("Scanning transactions..."):
            
            # Ensure we drop the target column if it accidentally exists
            if 'Class' in df.columns:
                X_live = df.drop('Class', axis=1)
            else:
                X_live = df.copy()

            try:
                # FIX 1: Automatically drop any text/string columns (keep only numbers)
                import numpy as np
                X_live = X_live.select_dtypes(include=[np.number])
                
                # FIX 2: Fill any missing/blank values with 0 to prevent NaN errors
                X_live = X_live.fillna(0)
                
                # Make predictions
                predictions = pipeline.predict(X_live)
                probabilities = pipeline.predict_proba(X_live)[:, 1]
                
                # Add results to the dataframe
                df['Fraud_Prediction'] = predictions
                df['Fraud_Probability (%)'] = (probabilities * 100).round(2)
                
                # Filter and display only the fraudulent transactions
                fraud_cases = df[df['Fraud_Prediction'] == 1]
                
                st.write("### Scan Results")
                if len(fraud_cases) > 0:
                    st.error(f"⚠️ Alert! Detected {len(fraud_cases)} potentially fraudulent transactions.")
                    st.dataframe(fraud_cases)
                else:
                    st.success("✅ All transactions appear legitimate. No fraud detected.")
                    
            except ValueError as e:
                # If it still fails, this will print the EXACT error to your Streamlit screen so you can debug it
                st.error(f"❌ Data Mismatch Error: {e}")
                st.info("Ensure your uploaded CSV has the exact same number of columns that your model was trained on.")
