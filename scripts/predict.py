# Importing required libraries
import joblib
import pandas as pd
import os

# Load model from outer directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.joblib')
try:
	model = joblib.load(model_path)
	print("Model successfully loaded.")
except Exception as e:
	print(f"Model could not be loaded: {e}")
	exit(1)



