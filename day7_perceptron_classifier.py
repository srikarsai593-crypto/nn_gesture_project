import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)

# --- Generate 2D two-class data ---
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=42)
# X shape: (200, 2) — two features per point
# y shape: (200,)   — 0 or 1 per point

# --- Perceptron: sigmoid neuron ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w, b):
    return sigmoid(np.dot(X, w) + b)

def bce_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- Training ---
w = np.zeros(2)   # one weight per feature (x1, x2)
b = 0.0
lr = 0.1
epochs = 500

for epoch in range(epochs):
    y_pred = predict(X, w, b)

    dw = np.dot(X.T, (y_pred - y)) / len(y)
    db = np.mean(y_pred - y)

    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        loss = bce_loss(y_pred, y)
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w}, b={b:.3f}")

print(f"\nFinal: w={w}, b={b:.3f}")

# --- Plot: scatter + decision boundary on a mesh grid ---
plt.figure(figsize=(8, 6))

# Create a mesh grid covering the data's range
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Predict the class for every point on the grid
grid_points = np.c_[xx1.ravel(), xx2.ravel()]  # flatten grid into (n,2) points
grid_preds = predict(grid_points, w, b).reshape(xx1.shape)

# Fill regions by predicted class (this IS the decision boundary, visually)
plt.contourf(xx1, xx2, grid_preds, levels=[0, 0.5, 1], colors=['#FFDDDD', '#DDDDFF'], alpha=0.6)
plt.contour(xx1, xx2, grid_preds, levels=[0.5], colors='black', linewidths=2)

# Overlay the actual data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', edgecolor='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', edgecolor='k')

plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary")
plt.savefig("day07_perceptron.png")
plt.show()