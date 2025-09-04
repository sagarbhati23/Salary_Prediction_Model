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

