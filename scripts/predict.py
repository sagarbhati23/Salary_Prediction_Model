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

# Load allowed values from CSV
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'expected_ctc.csv')
df = pd.read_csv(csv_path)
allowed_industries = set(df['Industry'].dropna().unique())
allowed_departments = set(df['Department'].dropna().unique())
allowed_roles = set(df['Role'].dropna().unique())
allowed_educations = set(df['Education'].dropna().unique())


# Fixed examples for prompts
industry_examples = "IT, Insurance, Retail"
department_examples = "Engineering, Sales, Education"
role_examples = "Manager, Scientist, Analyst, Team Lead"




