import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # Convert to MB for readability
    return memory_info.rss / (1024 * 1024)

# Load the C model metrics
try:
    c_metrics = pd.read_csv('./data/c_model_metrics.csv')
    c_metrics_dict = dict(zip(c_metrics['metric'], c_metrics['value']))
    print("Loaded C model metrics:")
    for metric, value in c_metrics_dict.items():
        print(f"  {metric}: {value}")
except FileNotFoundError:
    print("C model metrics file not found. Run the C program first.")
    c_metrics_dict = {}

# Now run the Python model and collect metrics
print("\nRunning Python logistic regression model...")

# Load the dataset from the CSV file
initial_memory = get_memory_usage()
df = pd.read_csv('./data/data_file.csv')

# Display the first few rows of the dataframe
print("Data Preview:")
print(df.head())

# Prepare the features and target variable
X = df[['X1', 'X2']]
y = df['Y']

# Memory usage after data loading
data_load_memory = get_memory_usage()
print(f"Memory usage after data loading: {data_load_memory:.2f} MB")
print(f"Memory increase for data loading: {data_load_memory - initial_memory:.2f} MB")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(
    C=1.0/0.1,  # Similar to L2 regularization in C code
    fit_intercept=True,
    max_iter=5000,
    tol=1e-6,
    verbose=1
)

# Measure training time
print("Training model...")
training_start_time = time.time()
training_start_memory = get_memory_usage()

model.fit(X_train, y_train)

training_end_time = time.time()
training_end_memory = get_memory_usage()

# Calculate and display metrics
py_training_time = training_end_time - training_start_time
py_training_memory = training_end_memory - training_start_memory

print(f"Training time: {py_training_time:.4f} seconds")
print(f"Memory usage during training: {py_training_memory:.2f} MB")

# Calculate and print model accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
py_accuracy = model.score(X, y)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")
print(f"Overall accuracy: {py_accuracy:.4f}")

# Prediction memory usage
prediction_start_memory = get_memory_usage()
y_pred = model.predict(X)
prediction_end_memory = get_memory_usage()
py_prediction_memory = prediction_end_memory - prediction_start_memory

# Store Python metrics
py_metrics_dict = {
    'training_time': py_training_time,
    'memory_usage_mb': py_training_memory + py_prediction_memory,
    'memory_init_mb': data_load_memory - initial_memory,
    'memory_training_mb': py_training_memory,
    'memory_prediction_mb': py_prediction_memory,
    'total_memory_mb': prediction_end_memory - initial_memory,
    'accuracy': py_accuracy,
    'intercept': model.intercept_[0] if model.fit_intercept else 0.0,
    'weight_1': model.coef_[0][0],
    'weight_2': model.coef_[0][1]
}

# Save Python metrics to CSV
py_metrics_df = pd.DataFrame(list(py_metrics_dict.items()), columns=['metric', 'value'])
py_metrics_df.to_csv('./data/python_model_metrics.csv', index=False)
print("\nPython metrics saved to ./data/python_model_metrics.csv")

# Create comparison visualization
plt.figure(figsize=(12, 10))

# Set up the plot
plt.subplot(2, 2, 1)
plt.title('Training Time Comparison')
if 'training_time' in c_metrics_dict:
    plt.bar(['C Implementation', 'Python Implementation'], 
            [c_metrics_dict['training_time'], py_metrics_dict['training_time']], 
            color=['blue', 'green'])
    plt.ylabel('Time (seconds)')
    for i, v in enumerate([c_metrics_dict['training_time'], py_metrics_dict['training_time']]):
        plt.text(i, v + 0.01, f"{v:.4f}s", ha='center')
else:
    plt.bar(['Python Implementation'], [py_metrics_dict['training_time']], color='green')
    plt.ylabel('Time (seconds)')
    plt.text(0, py_metrics_dict['training_time'] + 0.01, f"{py_metrics_dict['training_time']:.4f}s", ha='center')

plt.subplot(2, 2, 2)
plt.title('Memory Usage Comparison')
if 'memory_usage_mb' in c_metrics_dict:
    plt.bar(['C Implementation', 'Python Implementation'], 
            [c_metrics_dict['memory_usage_mb'], py_metrics_dict['memory_usage_mb']], 
            color=['blue', 'green'])
    plt.ylabel('Memory (MB)')
    for i, v in enumerate([c_metrics_dict['memory_usage_mb'], py_metrics_dict['memory_usage_mb']]):
        plt.text(i, v + 0.1, f"{v:.2f} MB", ha='center')
else:
    plt.bar(['Python Implementation'], [py_metrics_dict['memory_usage_mb']], color='green')
    plt.ylabel('Memory (MB)')
    plt.text(0, py_metrics_dict['memory_usage_mb'] + 0.1, f"{py_metrics_dict['memory_usage_mb']:.2f} MB", ha='center')

plt.subplot(2, 2, 3)
plt.title('Model Accuracy Comparison')
if 'accuracy' in c_metrics_dict:
    plt.bar(['C Implementation', 'Python Implementation'], 
            [c_metrics_dict['accuracy'] * 100, py_metrics_dict['accuracy'] * 100], 
            color=['blue', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 100])
    for i, v in enumerate([c_metrics_dict['accuracy'] * 100, py_metrics_dict['accuracy'] * 100]):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
else:
    plt.bar(['Python Implementation'], [py_metrics_dict['accuracy'] * 100], color='green')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 100])
    plt.text(0, py_metrics_dict['accuracy'] * 100 + 1, f"{py_metrics_dict['accuracy'] * 100:.2f}%", ha='center')

# Decision boundary plot
plt.subplot(2, 2, 4)
plt.title('Decision Boundaries')

# Create a mesh grid for plotting decision boundary
x_min, x_max = X['X1'].min() - .5, X['X1'].max() + .5
y_min, y_max = X['X2'].min() - .5, X['X2'].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Python model decision boundary
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# C model decision boundary (if we have the weights)
if all(k in c_metrics_dict for k in ['intercept', 'weight_1', 'weight_2']):
    # Manually calculate decision boundary for C model
    w1 = c_metrics_dict['weight_1']
    w2 = c_metrics_dict['weight_2']
    b = c_metrics_dict['intercept']
    Z_c = (np.c_[xx.ravel(), yy.ravel()] @ np.array([w1, w2]) + b) > 0
    Z_c = Z_c.reshape(xx.shape)
    plt.contour(xx, yy, Z_c, colors='blue', linewidths=2, levels=[0.5])

# Plot data points
plt.scatter(X['X1'], X['X2'], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
plt.xlabel('X1')
plt.ylabel('X2')

# Legend for the decision boundaries
if all(k in c_metrics_dict for k in ['intercept', 'weight_1', 'weight_2']):
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='C Model'),
        Line2D([0], [0], color='red', lw=2, label='Python Model')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

# Create metrics table below the plots
if c_metrics_dict:
    metrics_data = [
        ['Metric', 'C Implementation', 'Python Implementation'],
        ['Training Time (s)', f"{c_metrics_dict.get('training_time', 'N/A'):.4f}", f"{py_metrics_dict['training_time']:.4f}"],
        ['Memory Usage (MB)', f"{c_metrics_dict.get('memory_usage_mb', 'N/A'):.2f}", f"{py_metrics_dict['memory_usage_mb']:.2f}"],
        ['Accuracy (%)', f"{c_metrics_dict.get('accuracy', 'N/A')*100:.2f}", f"{py_metrics_dict['accuracy']*100:.2f}"],
    ]
else:
    metrics_data = [
        ['Metric', 'Python Implementation'],
        ['Training Time (s)', f"{py_metrics_dict['training_time']:.4f}"],
        ['Memory Usage (MB)', f"{py_metrics_dict['memory_usage_mb']:.2f}"],
        ['Accuracy (%)', f"{py_metrics_dict['accuracy']*100:.2f}"],
    ]

plt.tight_layout()
plt.figtext(0.5, 0.01, f"C vs Python Logistic Regression Performance Comparison", 
            ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

# Save the figure
plt.savefig('./data/model_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to ./data/model_comparison.png")

# Display the comparison table
fig, ax = plt.figure(figsize=(10, 3)), plt.subplot(111)
ax.axis('off')
table = ax.table(cellText=metrics_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.savefig('./data/metrics_table.png', dpi=300, bbox_inches='tight')
print("Metrics table saved to ./data/metrics_table.png")