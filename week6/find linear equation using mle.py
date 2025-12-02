import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
 
# Step 1: Graphing the Relationship
X = np.array([0.5, 2.3, 2.9]).reshape(-1, 1)
y = np.array([1.4, 1.9, 3.2]).reshape(-1, 1)
 
# Step 2: Applying the Derived Equation
def max_lik_estimate(X, y):
    theta_ml = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_ml
 
# Step 3: Plotting the Initial Model
def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta
    return prediction
 
theta_ml = max_lik_estimate(X, y)
print("Theta (ML Estimate):", theta_ml[0][0])
 
Xtest = np.linspace(-5, 5, 100).reshape(-1, 1)
ml_prediction = predict_with_estimate(Xtest, theta_ml)
 
# Step 4: Improving the Model
N, D = X.shape
X_aug = np.hstack([np.ones((N, 1)), X])
theta_ml_bias = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
print("Theta with Bias:", theta_ml_bias)
 
# Step 5: Plotting the Improved Model
improved_prediction = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest]) @ theta_ml_bias
y_pred = np.hstack([np.ones((X.shape[0], 1)), X]) @ theta_ml_bias
plt.figure()
plt.plot(X, y, '+', markersize=10, label='Data points')
plt.plot(Xtest, improved_prediction, label=f'y = {theta_ml_bias[1][0]:.2f}x + {theta_ml_bias[0][0]:.2f}')
plt.xlabel("X-axis ($x$)")
plt.ylabel("y-axis ($y$)")
plt.title("Plot of Training Data Set")
plt.xlim([-5, 5])
plt.legend()
plt.grid(True)
plt.show()
 
# RMSE Calculation
def RMSE(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))
 
rmse_test = RMSE(y, y_pred)
print("RMSE:", rmse_test)
 
# Loss Function
def loss_function(Phi, y, theta):
    return np.sum((y - Phi.dot(theta)) ** 2)
 
# Gradient Descent
def gradient_descent(Phi, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        prediction = Phi.dot(theta)
        error = prediction - y
        gradient = (1 / m) * Phi.T.dot(error)
        theta -= alpha * gradient
        cost_history[it] = loss_function(Phi, y, theta)
 
        # Early stopping if NaN or cost increases
        if np.isnan(cost_history[it]) or (it > 0 and cost_history[it] > cost_history[it - 1]):
            print(f"Breaking at iteration {it} due to NaN or cost increase")
            break
 
    return theta, cost_history[:it + 1]
 
# Gradient Descent Implementation
np.random.seed(41)
numberTheta = 2
theta = np.random.uniform(-1, 1, (numberTheta, 1))
alpha = 0.0002
iterations = 100000
 
theta_gd, cost_history = gradient_descent(X_aug, y, theta, alpha, iterations)
print("Theta (Gradient Descent):", theta_gd)
 
#RMSE
y_pred_gd = X_aug.dot(theta_gd)
rmse_gd = RMSE(y, y_pred_gd)
print("RMSE (Gradient Descent):", rmse_gd)
 
# Plotting Cost History
plt.figure()
plt.plot(range(len(cost_history)), cost_history, "b.")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.grid(True)
plt.show()