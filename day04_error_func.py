import numpy as np

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def binary_cross_entropy(y_pred, y_true):
    epsilon = 1e-15  # prevents log(0), which is -infinity
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Simulate y_true fixed, y_pred slowly "learning" toward it
y_true = np.array([1, 0, 1, 1, 0])

# Start with random, wrong-ish predictions
y_pred = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
print("Bad predictions:")
print("  MSE:", mse_loss(y_pred, y_true))
print("  BCE:", binary_cross_entropy(y_pred, y_true))

# Move predictions closer to the truth, step by step
for step in range(5):
    y_pred = y_pred + (y_true - y_pred) * 0.3  # nudge 30% closer to truth each step
    print(f"\nStep {step+1}, predictions: {np.round(y_pred, 3)}")
    print("  MSE:", round(mse_loss(y_pred, y_true), 5))
    print("  BCE:", round(binary_cross_entropy(y_pred, y_true), 5))