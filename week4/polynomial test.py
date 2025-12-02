import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

X = np.array([-4.5, -3.5, -3, -1.8, -0.2, 0.3, 1.3, 2.6, 3.8, 4.8]).reshape(-1,1) # 10x1 vector, N=5, D=1
y = np.array([
    [-1.11362822],
    [-1.24394281],
    [-0.91157385],
    [0.67067171],
    [1.24891634],
    [0.7776148],
    [-0.62067303],
    [-1.41641754],
    [-0.30383694],
    [0.92323755]
]).reshape(-1,1) # 10x1 vector

def max_lik_estimate(X, y):
    theta_ml = np.linalg.solve(X.T @ X, X.T @ y)
    return theta_ml

def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta
    return prediction

theta_ml = max_lik_estimate(X, y)
print(theta_ml[0][0])

Xtest = np.linspace(-5, 5, 100).reshape(-1,1)
ml_prediction = predict_with_estimate(Xtest, theta_ml)

# def f(x):
#     return np.cos(x) + 0.2 * np.random.normal(size=(x.shape))

# X= np.linspace(-4,4,20).reshape(-1,1)
# y = f(X)

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
# plt.plot(Xtest, ml_prediction, label=f'y = {theta_ml[0][0]}x')
plt.plot(Xtest, improved_prediction, label=f'y = {theta_ml_bias[0][0]}x')
plt.xlabel ("X-axis ($x$)")
plt.ylabel ("y-axis ($y$)")
plt.title ("Plot of Training Data Set")
plt.xlim([-5,5])
plt.legend()
plt.grid(True)
plt.show()
