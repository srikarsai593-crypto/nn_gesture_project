import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducible noise

# --- Generate data ---
X = np.random.uniform(-5, 5, 100)
noise = np.random.normal(0, 1, 100)
y = 2 * X + 1 + noise

# --- Neuron: linear activation ---
def predict(X, w, b):
    return w * X + b

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# --- Training setup ---
w, b = 0.0, 0.0   # start from scratch, network knows nothing
lr = 0.01
epochs = 200
snapshots = {}   # store (w, b) at chosen epochs

for epoch in range(epochs + 1):
    y_pred = predict(X, w, b)

    if epoch in [0, 10, 50, 200]:
        snapshots[epoch] = (w, b)
        loss = mse_loss(y_pred, y)
        print(f"Epoch {epoch}: w={w:.3f}, b={b:.3f}, loss={loss:.3f}")

    # Gradients
    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    # Update
    w -= lr * dw
    b -= lr * db

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='lightgray', label='Noisy data', s=20)

x_line = np.linspace(-5, 5, 100)
colors = {0: 'red', 10: 'orange', 50: 'green', 200: 'blue'}

for epoch, (w_snap, b_snap) in snapshots.items():
    y_line = predict(x_line, w_snap, b_snap)
    plt.plot(x_line, y_line, color=colors[epoch], label=f'Epoch {epoch}')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Neuron learning y = 2x + 1 from noisy data")
plt.grid(True)
plt.savefig("day07_linear_regression.png")
plt.show()