import numpy as np
import matplotlib.pyplot as plt

# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)  # NumPy has this built in, matches the formula exactly

def relu(x):
    return np.maximum(0, x)

# --- Derivatives ---
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# --- Plot all 6 in a 2x3 grid ---
x = np.linspace(-6, 6, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

functions = [
    (sigmoid(x), "Sigmoid"),
    (tanh(x), "Tanh"),
    (relu(x), "ReLU"),
    (sigmoid_derivative(x), "Sigmoid Derivative"),
    (tanh_derivative(x), "Tanh Derivative"),
    (relu_derivative(x), "ReLU Derivative"),
]

for ax, (y, title) in zip(axes.flat, functions):
    ax.plot(x, y)
    ax.set_title(title)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.grid(True)

plt.tight_layout()
plt.savefig("day03_activations.png")
plt.show()