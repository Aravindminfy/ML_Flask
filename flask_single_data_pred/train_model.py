# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load and clean data
df = pd.read_csv(r'C:\Users\Minfy.DESKTOP-3E50D5N\Documents\python_flask\Bank_Personal_Loan_Modelling.csv')  # Make sure CSV is in same folder
df.drop(['ID', 'ZIP Code'], axis=1, inplace=True)
df = df[df['Experience'] >= 0]

# Log transform
for col in ['Income', 'CCAvg', 'Mortgage']:
    df[col] = np.log1p(df[col])

df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns=['Experience', 'Mortgage'], inplace=True)

# Split
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])
pipeline.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, 'model/rf_pipeline.pkl')
print("✅ Model saved to model/rf_pipeline.pkl")
