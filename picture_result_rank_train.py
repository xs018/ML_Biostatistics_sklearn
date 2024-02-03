import matplotlib.pyplot as plt
import numpy as np

r2_scores = [
    0.96671049, 0.970162442, 0.976268564, 0.858973124, 0.858914146,
    0.965336504, 0.918515567, 0.959801488, 0.962400979, 0.969096311,
    0.960968465, 0.85893131, 0.854064759
]

adjusted_r2_scores = [
    0.965572387, 0.969142355, 0.975457232, 0.854151693, 0.854090698,
    0.964151428, 0.915729774, 0.95842718, 0.961115542, 0.968039775,
    0.959634054, 0.854108449, 0.84907552
]

# Define the model names in the same order as the provided scores
model_names = [
    "SVR", "GradientBoosting", "KNeighbors", "Ridge", "ElasticNetCV",
    "MLP", "AdaBoost", "DecisionTree", "ExtraTrees", "xgboost",
    "RandomForest", "LASSO", "Lightgbm"
]

# Sort the scores along with model names based on R2 scores in descending order
sorted_indices = np.argsort(r2_scores)[::-1]
sorted_model_names = np.array(model_names)[sorted_indices]
sorted_r2_scores = np.array(r2_scores)[sorted_indices]
sorted_adjusted_r2_scores = np.array(adjusted_r2_scores)[sorted_indices]

# Re-create the bar plot with the sorted values
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate bar positions
bar_width = 0.35
index = np.arange(len(sorted_model_names))

# Placing the bars for adjusted R2 slightly to the right
bar1 = ax.barh(index + bar_width, sorted_adjusted_r2_scores, bar_width, label='Adjusted R2', color='orange')
bar2 = ax.barh(index, sorted_r2_scores, bar_width, label='R2', color='royalblue')


# Set y-axis labels to sorted model names
ax.set(yticks=index + bar_width / 2, yticklabels=sorted_model_names)

# Add legend, title, and axis labels
ax.legend()
ax.set_title('R2 and Adjusted R2 Scores for Various Models in Train Dataset (Sorted by R2 Score)')
ax.set_xlabel('Score')

# Show the plot
plt.tight_layout()
plt.show()
