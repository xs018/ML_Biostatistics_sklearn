import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Provided data
data_test = {
    'Model': ['SVR', 'GBR', 'KN', 'MLP', 'ABR', 'ETR', 'XGB', 'RF'],
    'Test_MAE': [0.01634912, 0.022044889, 0.058692513, 0.023906684, 0.050138857, 0.048167133, 0.030054148, 0.043109503],
    'Test_MSE': [0.001714101, 0.001282933, 0.005335514, 0.001276616, 0.004675515, 0.003944987, 0.001906152, 0.003179511]
}

data_train = {
    'Model': ['SVR', 'GBR', 'KN', 'MLP', 'ABR', 'ETR', 'XGB', 'RF'],
    'Train_MAE': [0.017894673, 0.022100208, 0.010908197, 0.026406871, 0.05060622, 0.029843699, 0.023148099, 0.031717263],
    'Train_MSE': [0.001665225, 0.001492549, 0.001187106, 0.001733955, 0.004076056, 0.001880797, 0.00154588, 0.001952455]
}

# Convert to DataFrame
df_test = pd.DataFrame(data_test)
df_train = pd.DataFrame(data_train)

# Combine the training and testing data into a single DataFrame for the heatmap
df_combined = pd.concat([df_train.set_index('Model'), df_test.set_index('Model')], axis=1)

# Create a combined metric DataFrame for heatmap
# For each metric, we will calculate a normalized score for heatmap intensity
metrics = ['MAE', 'MSE']
combined_metrics = {}

for metric in metrics:
    train_metric = f'Train_{metric}'
    test_metric = f'Test_{metric}'
    combined_metrics[train_metric] = df_combined[train_metric].values
    combined_metrics[test_metric] = df_combined[test_metric].values

# Convert combined metrics to a DataFrame
df_metrics = pd.DataFrame(combined_metrics, index=df_combined.index)

# Normalize the metrics for better color mapping
# We will use a Min-Max scaling within each metric
for metric in df_metrics.columns:
    min_val = df_metrics[metric].min()
    max_val = df_metrics[metric].max()
    df_metrics[metric] = (df_metrics[metric] - min_val) / (max_val - min_val)

# Create a heatmap of the metrics
plt.figure(figsize=(10, 6))
sns.heatmap(df_metrics, annot=True, fmt=".2f", cmap="YlGn")
plt.title('Normalized MAE and MSE for Training and Testing Datasets')
plt.ylabel('Models')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
