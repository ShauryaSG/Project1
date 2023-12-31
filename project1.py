"""
Shaurya Ganguly
AER850
Project 1
Oct/15/2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score
import joblib


"""
data preprocessing and visualization
"""

# Step 1: Read the data from the CSV file and turn it into a dataframe
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


#Step 3: Correlation analysis
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



"""
machine learning part
"""


# Step 4: Data Preprocessing
x = df.drop('Step', axis=1)
y = df['Step']

#Split the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#create the models
logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()

# Step 5: Grid Search Cross-Validation for Hyperparameter Tuning

# Define hyperparameters for each model
logistic_params = {'C': [0.1, 1, 10]}
decision_tree_params = {'max_depth': [None, 10, 20]}
random_forest_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}

# Perform Grid Search for each model
logistic_grid = GridSearchCV(logistic_model, logistic_params, cv=5)
logistic_grid.fit(x_train, y_train)

decision_tree_grid = GridSearchCV(decision_tree_model, decision_tree_params, cv=5)
decision_tree_grid.fit(x_train, y_train)

random_forest_grid = GridSearchCV(random_forest_model, random_forest_params, cv=5)
random_forest_grid.fit(x_train, y_train)

# Print best hyperparameters for each model
print("Best Hyperparameters for Logistic Regression:", logistic_grid.best_params_)
print("Best Hyperparameters for Decision Tree:", decision_tree_grid.best_params_)
print("Best Hyperparameters for Random Forest:", random_forest_grid.best_params_)

#Step 6: Evaluate models
logistic_best_model = logistic_grid.best_estimator_
decision_tree_best_model = decision_tree_grid.best_estimator_
random_forest_best_model = random_forest_grid.best_estimator_

# Predictions
y_pred_logistic = logistic_best_model.predict(x_test)
y_pred_decision_tree = decision_tree_best_model.predict(x_test)
y_pred_random_forest = random_forest_best_model.predict(x_test)

# Print evaluation metrics for each model
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Classification Report:\n", classification_report(y_test, y_pred_logistic))

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, y_pred_decision_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_decision_tree))

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_random_forest))
print("Classification Report:\n", classification_report(y_test, y_pred_random_forest))


# Calculate metrics for each model
metrics = {
    'Logistic Regression': {
        'f1_score': f1_score(y_test, y_pred_logistic, average='weighted'),
        'precision': precision_score(y_test, y_pred_logistic, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred_logistic)
    },
    'Decision Tree': {
        'f1_score': f1_score(y_test, y_pred_decision_tree, average='weighted'),
        'precision': precision_score(y_test, y_pred_decision_tree, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred_decision_tree)
    },
    'Random Forest': {
        'f1_score': f1_score(y_test, y_pred_random_forest, average='weighted'),
        'precision': precision_score(y_test, y_pred_random_forest, average='weighted'),
        'accuracy': accuracy_score(y_test, y_pred_random_forest)
    }
}

# Display the metrics for each model
print("\nModel Performance Metrics:")
for model, scores in metrics.items():
    print(f"\n{model}:")
    print("F1 Score:", scores['f1_score'])
    print("Precision:", scores['precision'])
    print("Accuracy:", scores['accuracy'])

#Step 7: Select the best model based on the F1 score
best_model = max(metrics, key=lambda k: metrics[k]['f1_score'])

#Step 8: Create a confusion matrix for the selected best model
best_model_predictions = {
    'Logistic Regression': y_pred_logistic,
    'Decision Tree': y_pred_decision_tree,
    'Random Forest': y_pred_random_forest
}

conf_matrix = confusion_matrix(y_test, best_model_predictions[best_model])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for {best_model}')
plt.show()


#Step 9: train the best model based on the confusion matrix
best_model = RandomForestClassifier(n_estimators=100, max_depth=10)
best_model.fit(x, y)

# Save the trained model in a joblib format
joblib.dump(best_model, 'best_model.joblib')
loaded_model = joblib.load('best_model.joblib')


#Step 10: Predict the maintenance step for the provided coordinates
coordinates_to_predict = [[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875],[0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]]
predictions = loaded_model.predict(coordinates_to_predict)

# Print the predicted maintenance steps for the given coordinates
print("Predicted Maintenance Steps:")
for i, coord in enumerate(coordinates_to_predict):
    print(f"Coordinates {i+1}: {coord} => Predicted Step: {predictions[i]}")