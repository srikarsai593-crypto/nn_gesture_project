import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducible init

# --- XOR dataset ---
# Classic example of data that is NOT linearly separable —
# Day 7's single-layer perceptron cannot solve this
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- Two-layer network ---
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        # Small random init — keeps initial activations small,
        # avoids saturating sigmoid / dead ReLU right out of the gate
        self.W1 = np.random.randn(n_input, n_hidden) * 0.01
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.01
        self.b2 = np.zeros((1, n_output))

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)          # hidden layer: ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self.sigmoid(self.z2)      # output layer: sigmoid
        return self.out

# --- Build network and run forward pass ---
nn = NeuralNetwork(n_input=2, n_hidden=4, n_output=1)
output = nn.forward(X)

# --- Shape trace (today's deliverable: correct shapes) ---
print("Shape trace:")
print(f"X   : {X.shape}")
print(f"z1  : {nn.z1.shape}")
print(f"a1  : {nn.a1.shape}")
print(f"z2  : {nn.z2.shape}")
print(f"out : {nn.out.shape}")

print("\nPredictions (untrained — expect ~0.5 everywhere, no training yet):")
for xi, yi, pred in zip(X, y, output):
    print(f"input={xi}, true={yi[0]}, predicted={pred[0]:.4f}")

# --- Plot: decision surface of the untrained network ---
# Even though it hasn't learned yet, this shows the network CAN
# represent a non-linear boundary (unlike Day 7's straight line)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                      np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = nn.forward(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, probs, levels=50, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Network output (sigmoid)')

for xi, yi in zip(X, y):
    color = 'blue' if yi[0] == 1 else 'red'
    plt.scatter(xi[0], xi[1], color=color, edgecolor='black', s=150, zorder=5)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Day 8: Two-layer network forward pass on XOR (untrained)")
plt.grid(True, alpha=0.3)
plt.savefig("day08_forward_pass.png")
plt.show()