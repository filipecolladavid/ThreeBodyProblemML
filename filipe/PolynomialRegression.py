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

# Reading the data
df = pd.read_csv("../data/mlNOVA/mlNOVA/X_train.csv")

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

print(X)
print(y)

# Polynomial Regression model
print("Polynomial Regression")
# X_train, X_val, y_train, y_val = custom_train_test_split(X,y,test_size=0.999)
sample_fraction = 0.01
# Sample 1% of the unique trajectory IDs
unique_trajectory_ids = X['trajectory_id'].unique()
sampled_trajectory_ids = np.random.choice(unique_trajectory_ids, size=int(len(unique_trajectory_ids) * sample_fraction), replace=False)

# Filter the training data based on the sampled trajectory IDs
X_train_sampled = X[X['trajectory_id'].isin(sampled_trajectory_ids)]
y_train_sampled = y[y['trajectory_id'].isin(sampled_trajectory_ids)]

y_train_sampled = y.sample(frac=sample_fraction)
best_model, best_rmse = validate_poly_regression(X_train, y_train, X_val, y_val) 
print(f"Best model: {best_model}, Best RMSE: {best_rmse}")