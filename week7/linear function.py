import numpy as np
import matplotlib.pyplot as plt

# Linear activation function and its derivative
def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

# Initialize weights and bias
np.random.seed(42)
weights = np.array([0.5, 0.5, 0.5, 0.5])
bias = np.random.rand(1)

# Dataset
inputs = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]
])
targets = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1])

# Learning rate
learning_rate = 0.01
loss_history = []

# Training loop
for epoch in range(1000):
    total_error = 0
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)

    for i in indices:
        input_layer = inputs[i]
        target = targets[i]

        z = np.dot(input_layer, weights) + bias
        output = linear(z)
        error = 0.5 * (target - output) ** 2
        total_error += error

        dE_dy = output - target
        dy_dz = linear_derivative(z)
        dz_dw = input_layer
        dz_db = 1

        gradient_weights = dE_dy * dy_dz * dz_dw
        gradient_bias = dE_dy * dy_dz * dz_db

        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    loss_history.append(total_error / len(inputs))

    if epoch % 100 == 0:
        print(f"[Linear] Epoch {epoch}, Avg Loss: {total_error / len(inputs)}")

print("Final weights (Linear):", weights)
print("Final bias (Linear):", bias)

# Predictions
print("\nPredictions using Linear Activation:")
for i in range(len(inputs)):
    z = np.dot(inputs[i], weights) + bias
    output = linear(z)
    print(f"Input: {inputs[i]}, Predicted Output: {output[0]:.4f}, Actual Target: {targets[i]}")

# Plot Loss
plt.plot(range(1000), loss_history, label="Linear Activation")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Loss Over Epochs (Linear)")
plt.legend()
plt.show()
