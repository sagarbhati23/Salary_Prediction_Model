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

print("--- Salary Prediction ---")
user_input = {}
user_input['Industry'] = get_input(f"Enter Industry (e.g., {industry_examples}): ", str, allowed_industries)
user_input['Department'] = get_input(f"Enter Department (e.g., {department_examples}): ", str, allowed_departments)
user_input['Role'] = get_input(f"Enter Role/Designation (e.g., {role_examples}): ", str, allowed_roles)
    
# For education, keep the first allowed value as example
education_example = next(iter(allowed_educations)) if allowed_educations else ""
user_input['Education'] = get_input(f"Enter Education Level (e.g., {education_example}): ", str, allowed_educations)
user_input['Total_Experience'] = get_input("Enter Total Experience (years, decimal allowed): ", float, positive=True)
user_input['No_Of_Companies_worked'] = get_input("Enter Number of Companies Worked: ", int, positive=True)
user_input['Current_CTC'] = get_input("Enter Current CTC (decimal allowed): ", float, positive=True)

X_new = pd.DataFrame([user_input])
try:
    predicted_salary = model.predict(X_new)[0]
    print(f"\nPredicted Salary (CTC): {predicted_salary:.2f}")
except Exception as e:
        print(f"Prediction failed: {e}")


