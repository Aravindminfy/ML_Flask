from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
import pandas as pd
import numpy as np
import joblib
import os
import uuid

app = Flask(__name__)
model = joblib.load('model/rf_pipeline.pkl')  # Your saved pipeline

STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

def preprocess_bulk(df):
    # Log transform numerical features
    df['Income'] = np.log1p(df['Income'])
    df['CCAvg'] = np.log1p(df['CCAvg'])
    # Mortgage to binary HasMortgage
    df['HasMortgage'] = df['Mortgage'].apply(lambda x: 1 if x > 0 else 0)

    # Drop columns not used in model
    drop_cols = ['ID', 'ZIP Code', 'Experience', 'Mortgage']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    return df

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    try:
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        df = pd.read_csv(file)
        df_processed = preprocess_bulk(df.copy())
        preds = model.predict(df_processed)
        df['Prediction'] = preds

        file_id = uuid.uuid4().hex
        filename = f"predictions_{file_id}.csv"
        save_path = os.path.join(STATIC_DIR, filename)
        df.to_csv(save_path, index=False)

        # Redirect to the view page with file_id
        return redirect(url_for('view_predictions', file_id=file_id))
    except Exception as e:
        return f"Error processing file: {e}", 500

@app.route('/view_predictions/<file_id>', methods=['GET'])
def view_predictions(file_id):
    filename = f"predictions_{file_id}.csv"
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        abort(404, description="File not found")

    # Load csv to display as HTML table
    df = pd.read_csv(file_path)
    table_html = df.to_html(classes='table table-striped', index=False)

    return render_template('view_predictions.html', table_html=table_html, filename=filename)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    # Security: ensure filename is inside static folder only
    if '..' in filename or filename.startswith('/'):
        abort(400)
    return send_from_directory(STATIC_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
