import numpy as np

# Random input: 5 features
X = np.random.randn(5)
print("X shape:", np.shape(X))

# Random weight matrix: 5 inputs -> 3 neurons
W = np.random.randn(5, 3)
print("W shape:", np.shape(W))

# Bias: one per neuron
b = np.random.randn(3)
print("b shape:", np.shape(b))

# Layer output
Z = np.dot(X, W) + b
print("Z shape:", np.shape(Z))
print("Z values:", Z)