import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
df = pd.read_csv('HousePricePrediction.csv')

# 2. Select numerical and categorical features matching your model
numeric_features = [
    'MSSubClass', 'LotArea', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF'
]
categorical_features = ['MSZoning', 'LotConfig', 'BldgType', 'Exterior1st']

# Handle missing values
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
df[categorical_features] = df[categorical_features].fillna('Missing')

# Drop rows where target variable (SalePrice) is missing
df = df.dropna(subset=['SalePrice'])

# Separate features (X) and target (y)
X = df[numeric_features + categorical_features]
y = df['SalePrice']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 3. Save artifacts required by app.py
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("Model, scaler, and column definitions saved successfully!")

