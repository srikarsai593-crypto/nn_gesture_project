import numpy as np
import time

np.random.seed(42)

# --- Reuse Day 8's network ---
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
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
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = self.sigmoid(self.z2)
        return self.out


# =========================================================
# PART 1 — Batch of 100 random samples, shape trace
# =========================================================
nn = NeuralNetwork(n_input=2, n_hidden=4, n_output=1)

X_batch = np.random.randn(100, 2)  # 100 samples, 2 features each
output = nn.forward(X_batch)

print("Shape trace (batch of 100):")
print(f"X   : {X_batch.shape}")
print(f"z1  : {nn.z1.shape}")
print(f"a1  : {nn.a1.shape}")
print(f"z2  : {nn.z2.shape}")
print(f"out : {nn.out.shape}")
print()


# =========================================================
# PART 2 — Speed test: Python loop vs vectorised matrix op
# =========================================================
n_samples = 10_000
n_input, n_hidden, n_output = 2, 4, 1

X_big = np.random.randn(n_samples, n_input)
W1 = np.random.randn(n_input, n_hidden) * 0.01
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_output) * 0.01
b2 = np.zeros((1, n_output))

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Slow way: loop over each sample one at a time ---
start = time.time()

outputs_loop = np.zeros((n_samples, n_output))
for i in range(n_samples):
    xi = X_big[i:i+1]              # shape (1, 2), keep it 2D
    z1 = xi @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    out = sigmoid(z2)
    outputs_loop[i] = out

loop_time = time.time() - start

# --- Fast way: one matrix operation for all samples at once ---
start = time.time()

z1 = X_big @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
outputs_vectorised = sigmoid(z2)

vector_time = time.time() - start

# --- Results ---
print(f"Loop time       ({n_samples:,} samples): {loop_time:.4f} sec")
print(f"Vectorised time ({n_samples:,} samples): {vector_time:.4f} sec")
print(f"Speedup: {loop_time / vector_time:.1f}x faster")

# --- Sanity check: both methods should give (almost) the same answer ---
max_diff = np.max(np.abs(outputs_loop - outputs_vectorised))
print(f"\nMax difference between loop and vectorised outputs: {max_diff:.2e}")