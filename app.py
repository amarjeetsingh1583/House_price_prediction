from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load saved assets
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract numeric fields
    input_data = {
        'MSSubClass': float(request.form['MSSubClass']),
        'LotArea': float(request.form['LotArea']),
        'OverallCond': float(request.form['OverallCond']),
        'YearBuilt': float(request.form['YearBuilt']),
        'YearRemodAdd': float(request.form['YearRemodAdd']),
        'BsmtFinSF2': float(request.form['BsmtFinSF2']),
        'TotalBsmtSF': float(request.form['TotalBsmtSF'])
    }

    # Handle One-Hot Encoded Categorical Selects
    for col in ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']:
        selected_val = request.form.get(col)
        encoded_col = f"{col}_{selected_val}"
        if encoded_col in model_columns:
            input_data[encoded_col] = 1

    # Create DataFrame aligned with original training columns
    df_input = pd.DataFrame([input_data])
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # Scale and predict
    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]

    return render_template('index.html', prediction_text=f'Estimated House Price: ${prediction:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)




