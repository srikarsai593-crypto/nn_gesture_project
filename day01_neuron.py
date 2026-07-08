import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))
def neuron(x,w,b):
    return sigmoid(np.dot(w,x)+b)

x = np.array([1.0, 4.0])
w = np.array([2.0, 3.0])
b = -5.0
output = neuron(x, w, b)
print("Neuron output:", output)

x_vals=np.linspace(-6,6,200)
y_vals=sigmoid(x_vals)

plt.plot(x_vals, y_vals)
plt.title("Sigmoid Activation Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.grid(True)
plt.savefig("day01_sigmoid.png")
plt.show()
