import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import statsmodels.tools.eval_measures as em
import os
from easy_mpl import regplot, plot
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns

##environment_python3.9.0(pytorch2xconda)

file_path = 'ML_photocatalysis2.xlsx'

# Read the Excel file
data = pd.read_excel(file_path, header=0)

# Assuming 'df' is your pandas DataFrame.
df = data.sample(frac=1, random_state=38).reset_index(drop=True)

# Separate the features and the target variable
X = df.drop('SA removal rate', axis=1)
y = df['SA removal rate']

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

sns.heatmap(X.corr(), cmap='Blues')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

## grid search tune the hyper-parameters
# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': [1000],
    'subsample': [0.50],
    'min_samples_split': [20],
    'min_samples_leaf': [2],
    'max_depth': [3, 5],
    'alpha': [0.1]
}

# Initialize the GradientBoostingRegressor model
reg = GradientBoostingRegressor(random_state=0)

# Initialize the Grid Search model
grid_search = GridSearchCV(reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model_GBR = grid_search.best_estimator_

# Predictions
y_pred = best_model_GBR.predict(X_test_scaled)

# Evaluate the model
n = X_test_scaled.shape[0]  # Number of observations
p = X_test_scaled.shape[1]  # Number of predictors

test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
test_adjusted_r_squared = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

print(f'Best model parameters: {grid_search.best_params_}')
print(f'MSE_test: {test_mse}, R2_test: {test_r2}, Adjust_R2_test: {test_adjusted_r_squared}')
y_train_pred = best_model_GBR.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
n = X_train_scaled.shape[0]  # Number of observations
p = X_train_scaled.shape[1]  # Number of predictors
train_adjusted_r_squared = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
print(f'MSE_train: {train_mse}, R2_train: {train_r2}, Adjust_R2_train: {train_adjusted_r_squared}')

n = len(y_train_pred)
def calculate_log_likelihood_from_mse(mse, n):
    llf = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(mse) - 0.5 * n
    return llf
log_likelihood = calculate_log_likelihood_from_mse(train_mse, n)
llf = log_likelihood
nobs = n
df_modelwc = len(X_train.columns)
bic_value = em.bic(llf, nobs, df_modelwc)
print(f'trainThe value of bic_test is: {bic_value}')

n = len(y_pred)
def calculate_log_likelihood_from_mse(mse, n):
    llf = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(mse) - 0.5 * n
    return llf
log_likelihood = calculate_log_likelihood_from_mse(test_mse, n)
llf = log_likelihood
nobs = n
df_modelwc = len(X_test.columns)
bic_value = em.bic(llf, nobs, df_modelwc)
print(f'testThe value of bic_test is: {bic_value}')

## regression plot

ax = regplot(y_train, y_train_pred, line_color='orange',
             marker_color='orange', marker_size=35, fill_color='orange',
             scatter_kws={'edgecolors':'black', 'linewidth':0.8, 'alpha': 0.8},
             show=False, label="Train")
_ = regplot(y_test, y_pred, line_color='royalblue', ax=ax,
                marker_color='royalblue', marker_size=35, fill_color='royalblue',
             scatter_kws={'edgecolors':'black', 'linewidth':0.8, 'alpha': 0.8},
             show=False, label="Test", ax_kws=dict(legend_kws=dict(loc=(0.1, 0.8))))
plt.show()

## Grid search to generate new datasets
import numpy as np
import pandas as pd
from itertools import product

# Define the columns as numpy arrays
column1 = np.arange(0, 10.1, 0.1)
column2 = np.arange(0, 2.6, 0.1)
column3 = np.arange(0, 1.6, 0.1)
column4 = np.arange(3, 5.1, 0.1)

# Create all combinations of the four columns
all_combinations = list(product(column1, column2, column3, column4))

# Convert to a pandas DataFrame
df = pd.DataFrame(all_combinations, columns=['Column 1', 'Column 2', 'Column 3', 'Column 4'])

df_y = best_model_GBR.predict(df)
df_y = pd.DataFrame(df_y)

# Define the path for the Excel file
excel_filename = 'df_y.xlsx'

# Save the DataFrame to an Excel file
df_y.to_excel(excel_filename, index=False)
# Assuming df is your primary DataFrame and df_y is the data you want to add as a new column
if isinstance(df_y, np.ndarray):
    df_y = pd.DataFrame(df_y)

df['y'] = df_y.iloc[:,0]  

print(df)
df = df[df['y'] <= 1]

# Display the DataFrame after the operation
max_y_index = df['y'].idxmax()

# Retrieve the row with the largest 'y' value
row_with_max_y = df.loc[max_y_index]

# Display the row
# print(row_with_max_y)


## Output the feature importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
feature_names = X.columns
# Assuming `best_model_MLP` is your trained MLPRegressor model and `X_test_scaled` is your scaled test dataset
result = permutation_importance(best_model_GBR, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get the importance of each feature
feature_importance = result.importances_mean

# Now let's plot this information
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create a bar plot
sns.barplot(x=feature_importance, y=feature_names, palette="GnBu_d")

# Add chart labels and title
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance using GradientBoostingRegressor')

plt.show()
