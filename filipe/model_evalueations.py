# Reading the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from model_utils import custom_train_test_split, evaluate_model, validate_poly_regression

df = pd.read_csv("C:/Users/Diogo Almeida/Desktop/AA_three-body_problem/data/mlNOVA/mlNOVA/X_train.csv")

# Cleaning the data

# Create the trajectory_id column
df['trajectory_id'] = df['Id'] // 257
# Filter out rows where all columns except 'Id' and trajectory_Id are zero
df_filtered = df.loc[~(df.drop(columns=['Id','trajectory_id']) == 0).all(axis=1)]

# Features
# Group by 'trajectory_id' and get the first occurrence of each group
first_occurrence = df_filtered.groupby('trajectory_id').first().reset_index()
# Merge first occurrence back to the original dataframe, only replacing columns that need to be kept constant
columns_to_replace = df_filtered.columns.difference(['t', 'trajectory_id', 'Id'])  # Columns to replace except 't' and 'trajectory_id'
# We merge 'first_occurrence' on 'trajectory_id' with the original dataframe,
# and only replace the required columns.
X_raw = df_filtered[['t', 'trajectory_id', 'Id']].merge(
    first_occurrence[['trajectory_id'] + list(columns_to_replace)],
    on='trajectory_id',
    how='left'
)
# Reorder the columns
X_raw = X_raw[['t', 'x_1', 'y_1', 'v_x_1', 'v_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3', 'Id', 'trajectory_id']]

# Target
Y_raw = df_filtered[['x_1', 'y_1', 'v_x_1', 'v_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3', 'Id', 'trajectory_id']]

# Permanent test data
# First, remove 10% of the data to use as a fixed test set
X_train_val, X_test, y_train_val, y_test = custom_train_test_split(X_raw, Y_raw, test_size=0.1, drop=False)
X_test.drop(columns=['trajectory_id', 'Id'])
y_test.drop(columns=['trajectory_id', 'Id'])

X = X_train_val
y = y_train_val

# print(X)
# print(y)


# Test multiple linear regressions
print("Normal linear regression")
X_train, X_val, y_train, y_val = custom_train_test_split(X,y)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression()) 
])
evaluate_model(pipeline, X_train, X_val, y_train, y_val)

print()

print("Regularization")
print("Ridge regression")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=10))
])
evaluate_model(pipeline, X_train, X_val, y_train, y_val)

print("Lasso regression")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=5))
])
evaluate_model(pipeline, X_train, X_val, y_train, y_val)

# Polynomial Regressions

unique_ids = X['trajectory_id'].unique()

# Calculate 1% of the dataset
n_1_percent = max(1, int(len(unique_ids) * 0.01))

sampled_ids = X['trajectory_id'].drop_duplicates().sample(n=n_1_percent).tolist()
print(sampled_ids)

# Select the 1% by maintaining the same trajectory_id
X_sampled = X[X['trajectory_id'].isin(sampled_ids)]
y_sampled = y[y['trajectory_id'].isin(sampled_ids)]

print(X_sampled.shape)
print(y_sampled.shape)

# Split the sampled data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_sampled, y_sampled, test_size=0.2)
best_model, best_rmse = validate_poly_regression(X_train, y_train, X_val, y_val) 
print(f"Best model: {best_model}, Best RMSE: {best_rmse}")