import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'rf_pipeline.pkl')
    return joblib.load(model_path)

def preprocess_bulk(df):
    df = df.copy()
    df['Income'] = np.log1p(df['Income'])
    df['CCAvg'] = np.log1p(df['CCAvg'])
    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)
    drop_cols = ['ID', 'ZIP Code', 'Experience', 'Mortgage']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("📁 Bulk Personal Loan Prediction")
    st.write("Upload a CSV file with the required columns and get predictions for each record.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())

            model = load_model()

            df_processed = preprocess_bulk(df)
            preds = model.predict(df_processed)
            df['Prediction'] = preds

            st.write("### Predictions")
            st.dataframe(df)

            csv_data = convert_df_to_csv(df)
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
