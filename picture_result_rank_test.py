import matplotlib.pyplot as plt
import numpy as np

# Define the model names and corresponding R2 and Adjusted R2 scores for test sets
model_names = [
    "SVR", "GradientBoosting", "KNeighbors", "Ridge", "ElasticNetCV",
    "MLP", "AdaBoost", "DecisionTree", "ExtraTrees", "xgboost",
    "RandomForest", "LASSO", "Lightgbm"
]

r2_scores = [
    0.9699335, 0.977496483, 0.906411481, 0.860700722, 0.860537434,
    0.977607285, 0.917988306, 0.893835249, 0.93080225, 0.966564801,
    0.944229221, 0.860672687, 0.836419333
]

adjusted_r2_scores = [
    0.965307885, 0.974034404, 0.892013247, 0.839270064, 0.839081655,
    0.974162251, 0.905371122, 0.877502211, 0.920156443, 0.961420925,
    0.935649101, 0.839237716, 0.811253076
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

bar1 = ax.barh(index + bar_width, sorted_adjusted_r2_scores, bar_width, label='Adjusted R2', color='orange')
bar2 = ax.barh(index, sorted_r2_scores, bar_width, label='R2', color='royalblue')

# Set y-axis labels to sorted model names
ax.set(yticks=index + bar_width / 2, yticklabels=sorted_model_names)

# Add legend, title, and axis labels
ax.legend()
ax.set_title('R2 and Adjusted R2 Scores for Various Models in Test Dataset (Sorted by R2 Score)')
ax.set_xlabel('Score')

# Show the plot
plt.tight_layout()
plt.show()
