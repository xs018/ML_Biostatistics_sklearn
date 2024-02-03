import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided data
data_test = {
    'Model': ['SVR', 'GBR', 'KN', 'MLP', 'ABR', 'ETR', 'XGB', 'RF'], 
    'Test_BIC': [-95.72471653, -104.7066559, -60.52433654, -104.8596689, 
                 -64.61775843, -69.88445712, -92.43259071, -76.57172368]
}

data_train = {
    'Model': ['SVR', 'GBR', 'KN', 'MLP', 'ABR', 'ETR', 'XGB', 'RF'],
    'Train_BIC': [-415.0939294, -428.4498073, -456.3838216, -410.159664, 
                  -305.8832305, -400.2421754, -424.1666773, -395.6803664]
}

# Convert to DataFrame
df_test = pd.DataFrame(data_test)
df_train = pd.DataFrame(data_train)

# Combine the training and testing BIC into a single DataFrame for the heatmap
df_bic = pd.concat([df_train.set_index('Model'), df_test.set_index('Model')], axis=1)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(df_bic, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title('BIC Scores for Training and Testing Datasets')
plt.ylabel('Models')
plt.xlabel('BIC Scores')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
