import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def f_prime(x):
    return 2*x

def gradient_descent(start_x, lr, steps=15):
    x = start_x
    path = [x]
    for _ in range(steps):
        x = x - lr * f_prime(x)
        path.append(x)
    return path

learning_rates = [0.01, 0.1, 1.0]

# Parabola for the background curve
x_curve = np.linspace(-6, 6, 200)
y_curve = f(x_curve)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, lr in zip(axes, learning_rates):
    path = gradient_descent(start_x=5, lr=lr, steps=15)
    path_y = [f(x) for x in path]

    ax.plot(x_curve, y_curve, color='lightgray', linewidth=2)  # the parabola
    ax.plot(path, path_y, 'o-', color='crimson', markersize=4)  # the descent path
    ax.set_title(f"lr = {lr}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)

    print(f"\nlr={lr}, path of x:")
    print(np.round(path, 4))

plt.tight_layout()
plt.savefig("day05_gradient_descent.png")
plt.show()