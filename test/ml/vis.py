import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# Read the CSV data (assuming it's saved as 'data.csv')
# If your data doesn't have headers, we'll add them
column_names = ['x1', 'x2', 'output']
data = pd.read_csv(r'data_file.csv', header=None, names=column_names)

# Extract features and target
X = data[['x1', 'x2']].values
y = data['output'].values

# Create a scatter plot
plt.figure(figsize=(10, 8))

# Create a custom colormap for better visibility
colors = ListedColormap(['#FF9999', '#66B2FF'])
markers = ['o', '^']
labels = ['Class 0', 'Class 1']

# Plot each class
for i, label in enumerate([0, 1]):
    plt.scatter(
        X[y == label, 0], 
        X[y == label, 1],
        c=[colors(i)],
        marker=markers[i],
        label=labels[i],
        edgecolor='black',
        s=100
    )

# Add decision boundary (if you want to visualize a classifier)
# This is an optional step - commented out for now
"""
# Create a mesh grid to visualize decision boundary
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Train a classifier (e.g., logistic regression)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)

# Plot the decision boundary
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=colors)
"""

plt.xlabel('Feature 1 (x1)')
plt.ylabel('Feature 2 (x2)')
plt.title('Binary Classification Data Visualization')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(r'classification_plot.png', dpi=300, bbox_inches='tight')