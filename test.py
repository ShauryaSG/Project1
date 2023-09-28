#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn as sk
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module

#import data
df = pd.read_csv("data.csv")

#plot the 3D scatter plot using Matplotlib
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(projection='3d')

ax.scatter(df['X'], df['Y'], df['Z'],s=100, color='red', edgecolor='black')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot')

plt.show()



"""
# Data analysis: Calculate summary statistics
summary_stats = df.groupby('Step').describe()

# Data visualization: Create scatter plots
plt.figure(figsize=(10, 6))
#plt.scatter(x,y,z,'Step')
plt.title('Coordinates at Different Steps')
plt.xlabel('Step')
plt.ylabel('Coordinate')
#plt.legend(['', ''])
plt.grid()
plt.show()

print("\nSummary Statistics:")
print(summary_stats)
"""