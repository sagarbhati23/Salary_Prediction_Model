# Model Evaluation Script for Salary Prediction
# This script evaluates the trained salary prediction model using test data.

import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths

# Set model path to models directory inside scripts folder
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'model.joblib')
TEST_DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'expected_ctc.csv')

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("Model successfully loaded.")
except Exception as e:
    print(f"Model could not be loaded: {e}")
    exit(1)

# Load test data
try:
    df = pd.read_csv(TEST_DATA_PATH)
    print("Test data loaded.")
except Exception as e:
    print(f"Test data could not be loaded: {e}")
    exit(1)

# Features used for prediction
features = [
    'Industry',
    'Department',
    'Role',
    'Education',
    'Total_Experience',
    'No_Of_Companies_worked',
    'Current_CTC'
]

target = 'Expected_CTC'

# Check if all required columns exist
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    print(f"Missing columns in test data: {missing}")
    exit(1)

# Prepare test data
X_test = df[features]
y_test = df[target]

# Predict
try:
    y_pred = model.predict(X_test)
except Exception as e:
    print(f"Prediction failed: {e}")
    exit(1)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R^2 Score: {r2:.4f}")
