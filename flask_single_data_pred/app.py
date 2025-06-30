# app.py

from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/rf_pipeline.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        input_data = {
            'Age': int(request.form['Age']),
            'Income': float(request.form['Income']),
            'Family': int(request.form['Family']),
            'CCAvg': float(request.form['CCAvg']),
            'Education': int(request.form['Education']),
            'Securities Account': int(request.form['Securities_Account']),
            'CD Account': int(request.form['CD_Account']),
            'Online': int(request.form['Online']),
            'CreditCard': int(request.form['CreditCard'])
        }

        # Feature engineering
        income_log = np.log1p(input_data['Income'])
        ccavg_log = np.log1p(input_data['CCAvg'])

        # Reconstructed input
        input_df = pd.DataFrame([{
            'Age': input_data['Age'],
            'Income': income_log,
            'Family': input_data['Family'],
            'CCAvg': ccavg_log,
            'Education': input_data['Education'],
            'Securities Account': input_data['Securities Account'],
            'CD Account': input_data['CD Account'],
            'Online': input_data['Online'],
            'CreditCard': input_data['CreditCard'],
            'HasMortgage': 0  # set default to 0; can be updated in form later
        }])

        prediction = model.predict(input_df)[0]
        result = '✅ Likely to accept Personal Loan' if prediction == 1 else '❌ Not likely to accept Personal Loan'

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
