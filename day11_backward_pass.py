import numpy as np

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.W1 = np.random.randn(n_input, n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * 0.5
        self.b2 = np.zeros((1, n_output))

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)   # a2 = ŷ, the network's output
        return self.a2

    def compute_loss(self, y_pred, y_true):
        # Mean squared error, with the 1/2 factor for clean gradients
        return np.mean(0.5 * (y_pred - y_true) ** 2)

    def backward(self, y_true):
        """
        Output layer gradients ONLY — dW1, db1 not implemented yet (later days).
        """
        m = self.X.shape[0]  # batch size

        # --- Output delta ---
        # dL/da2 = (a2 - y)              <- from L = 1/2 (a2-y)^2
        # da2/dz2 = a2 * (1 - a2)         <- sigmoid's local gradient
        sigmoid_prime = self.a2 * (1 - self.a2)
        delta2 = (self.a2 - y_true) * sigmoid_prime      # shape (m, n_output)

        # --- Weight gradient for W2 ---
        # z2 = a1 @ W2 + b2  ->  local gradient of z2 w.r.t. W2 is a1
        # dL/dW2 = a1.T @ delta2 , averaged over the batch
        dW2 = (self.a1.T @ delta2) / m                   # shape (n_hidden, n_output)
        db2 = np.sum(delta2, axis=0, keepdims=True) / m  # shape (1, n_output)

        # Store so we can check them / use later
        self.dW2 = dW2
        self.db2 = db2
        self.delta2 = delta2

        return dW2, db2


# =========================================================
# Numerical gradient check
# =========================================================
def numerical_gradient_check(nn, X, y, epsilon=1e-5):
    """
    Compares analytic dW2 (from backward()) against the numerical
    gradient: (L(w+eps) - L(w-eps)) / (2*eps), computed one weight at a time.
    """
    # --- Run forward + backward once to get the analytic gradient ---
    nn.forward(X)
    dW2_analytic, db2_analytic = nn.backward(y)

    print("Checking dW2 (weights, output layer):")
    dW2_numeric = np.zeros_like(nn.W2)

    it = np.nditer(nn.W2, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        original_value = nn.W2[idx]

        # L(w + eps)
        nn.W2[idx] = original_value + epsilon
        y_pred_plus = nn.forward(X)
        loss_plus = nn.compute_loss(y_pred_plus, y)

        # L(w - eps)
        nn.W2[idx] = original_value - epsilon
        y_pred_minus = nn.forward(X)
        loss_minus = nn.compute_loss(y_pred_minus, y)

        # Restore original value
        nn.W2[idx] = original_value

        # Numerical gradient formula
        grad_numeric = (loss_plus - loss_minus) / (2 * epsilon)
        dW2_numeric[idx] = grad_numeric

        it.iternext()

    # --- Compare analytic vs numeric ---
    diff = np.abs(dW2_analytic - dW2_numeric)
    max_diff = np.max(diff)

    print(f"Analytic dW2:\n{dW2_analytic}")
    print(f"Numeric  dW2:\n{dW2_numeric}")
    print(f"Max absolute difference: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("✅ PASSED — gradients match within tolerance.")
    else:
        print("❌ FAILED — bug somewhere in the backward() formula.")


# --- Run everything ---
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(n_input=2, n_hidden=4, n_output=1)
numerical_gradient_check(nn, X, y)