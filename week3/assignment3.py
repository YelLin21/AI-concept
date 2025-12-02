import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#Step 1: Graphing the Relationship
X = np.array([-3, -1, 0.0, 1, 3]).reshape(-1, 1)
y = np.array([2.2, 3.7, 3.14, 3.67, 4.67]).reshape(-1, 1) 

#Step 2: Applying the Derived Equation
def max_lik_estimate(X, y):
    theta_ml = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_ml

#Step 3: Plotting the Initial Model
def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta
    return prediction

theta_ml = max_lik_estimate(X, y)
print(theta_ml[0][0])

Xtest = np.linspace(-5, 5, 100).reshape(-1,1)
ml_prediction = predict_with_estimate(Xtest, theta_ml)

#Step 4: Step 4: Improving the Model
print("="*10)
N, D = X.shape
X_aug = np.hstack([np.ones((N, 1)), X])
print("X_aug = ")
print(X_aug)
print("="*10)
# Recompute Theta
theta_ml_bias = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
print( theta_ml_bias)

#Step 5: Plotting the Improved Model
improved_prediction = np.hstack([np.ones((Xtest.shape[0], 1)), Xtest]) @ theta_ml_bias
plt.figure()
plt.plot(X, y, '+', markersize = 10 , label = 'Data points')
plt.plot(Xtest, ml_prediction, label=f'y = {theta_ml[0][0]}x')
plt.plot(Xtest, improved_prediction, label=f'y = {theta_ml_bias[0][0]}x')
plt.xlabel ("X-axis ($x$)")
plt.ylabel ("y-axis ($y$)")
plt.title ("Plot of Training Data Set")
plt.xlim([-5,5])
plt.legend()
plt.grid(True)
plt.show()

#Step 6: Analytical explanation
print("Analytical explanation: ")
print("Step 2 Limitation: Assumes the line passes through the origin (y=θx), "
      "which may not fit real-world data where y-intercept isn’t zero."
	"Bias Term Role: Adding ones enables the model to estimate a non-zero intercept (y=θ0+θ1x), making it more flexible."
	"Improved Fit: Accounts for vertical shifts, reducing residual errors and better aligning the line with the data."
	"Broader Model Capability: Supports relationships with both slope and intercept, fitting a wider range of linear patterns."
	"Mathematical Impact: Minimizes errors by recalculating parameters (θ0 and θ1) for a closer match to data.")
