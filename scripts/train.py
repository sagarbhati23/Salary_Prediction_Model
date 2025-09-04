import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
data_path = os.path.join('..', 'data', 'expected_ctc.csv')
df = pd.read_csv(data_path)

# feature columns (as per the dataset)
features = [
    'Industry', 'Department', 'Role', 'Total_Experience', 'No_Of_Companies_worked', 'Current_CTC', 'Education'
]
target = 'Expected_CTC'

# Split features into categorical and numerical
categorical_features = ['Industry', 'Department', 'Role', 'Education']
numerical_features = ['Total_Experience', 'No_Of_Companies_worked', 'Current_CTC']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
X = df[features]
y = df[target]
pipeline.fit(X, y)

