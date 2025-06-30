import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load data
    df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

    # Data preprocessing (same as your EDA script)
    df = df[df['Experience'] >= 0]
    df['Income'] = np.log1p(df['Income'])
    df['CCAvg'] = np.log1p(df['CCAvg'])
    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['ID', 'ZIP Code', 'Experience', 'Mortgage'], inplace=True)

    # Split features and target
    X = df.drop(columns=['Personal Loan'])
    y = df['Personal Loan']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make sure model directory exists
    os.makedirs('model', exist_ok=True)

    # Save the pipeline
    joblib.dump(pipeline, 'model/rf_pipeline.pkl')
    print("Model training completed and saved to model/rf_pipeline.pkl")

if __name__ == "__main__":
    main()
