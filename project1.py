import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the data from the CSV file
df = pd.read_csv('data.csv')

# Step 2: Plot the 3D scatter plot using Matplotlib, color by 'Step'
fig = plt.figure(figsize=(15, 13))
ax = fig.add_subplot(projection='3d')

# Scatter plot with different colors for each 'Step'
unique_steps = df['Step'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_steps)))

for i, step in enumerate(unique_steps):
    step_data = df[df['Step'] == step]
    ax.scatter(step_data['X'], step_data['Y'], step_data['Z'], label=f'Step {step}', c=[colors[i]], marker='o')

# Display details
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot')
ax.legend()
plt.show()

# Data analysis: Calculate summary statistics
print("\nSummary Statistics:")
print(df.groupby('Step').describe())


# Correlation analysis
step = 'Step'

# Calculate Pearson correlation coefficients
correlations = df.corr(method='pearson')
correlations_with_target = correlations[step]

# Generate a correlation plot
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Pearson Correlation Heatmap')
plt.show()

# Explain the correlation with the target variable
print("Correlation of features with the target variable:\n")
print(correlations_with_target)
# Provide explanation of the correlations and their impact on predictions
print("\nExplanation of correlations:")
print("A positive correlation with the target variable indicates that as the feature increases, the target variable tends to increase as well. Conversely, a negative correlation suggests an inverse relationship, where as the feature increases, the target variable tends to decrease. The magnitude of correlation, close to 1 or -1, indicates the strength of the relationship between the feature and the target variable. Understanding these correlations is crucial for making predictions using the features.")

