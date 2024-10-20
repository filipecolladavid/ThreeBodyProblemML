# Reading the data
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from model_utils import custom_train_test_split, train_evaluate_model, validate_poly_regression

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
print(X_raw)

# Target
Y_raw = df_filtered[['x_1', 'y_1', 'v_x_1', 'v_y_1', 'x_2', 'y_2', 'v_x_2', 'v_y_2', 'x_3', 'y_3', 'v_x_3', 'v_y_3', 'Id', 'trajectory_id']]
print(Y_raw)

# Permanent test data
# First, remove 10% of the data to use as a fixed test set
X_train_val, X_test, y_train_val, y_test = custom_train_test_split(X_raw, Y_raw, test_size=0.1, drop=False)
X_test.drop(columns=['trajectory_id', 'Id'])
y_test.drop(columns=['trajectory_id', 'Id'])

X = X_train_val
y = y_train_val

# Test multiple linear regressions - 
print("Normal linear regression")
X_train, X_val, y_train, y_val = custom_train_test_split(X, y, drop=True, columns_to_drop=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'])
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression()) 
])
train_evaluate_model(pipeline, X_train, X_val, y_train, y_val)

print()

print("Regularizations")
print("Ridge regression")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=10))
])
train_evaluate_model(pipeline, X_train, X_val, y_train, y_val)

print("Lasso regression")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Lasso(alpha=5))
])
train_evaluate_model(pipeline, X_train, X_val, y_train, y_val)

# Polynomial Regressions

# Get one percent of data - too many features cannot be fitted 
X_train, _, y_train, _  = custom_train_test_split(X, y, 0.99, False)
X_train, X_val, y_train, y_val,  = custom_train_test_split(X_train, y_train, drop=True, columns_to_drop=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'])
print(X_train.shape)
print(y_train.shape)

# Standard regressor
best_model, best_rmse = validate_poly_regression(X_train, y_train, X_val, y_val, degrees=range(1,10), plot=True) 
print(f"Best model: {best_model}, Best RMSE: {best_rmse}")

# Ridge
best_model, best_rmse = validate_poly_regression(X_train, y_train, X_val, y_val, degrees=range(1,10), regressor=Ridge()) 
print(f"Best model: {best_model}, Best RMSE: {best_rmse}")

# Lasso
best_model, best_rmse = validate_poly_regression(X_train, y_train, X_val, y_val, degrees=range(1,10), regressor=Lasso()) 
print(f"Best model: {best_model}, Best RMSE: {best_rmse}")