import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

X = np.array([-3,-1, 0.0,1,3]).reshape(-1,1)
y = np.array([-1.2, -0.7, 0.14,0.67,1.67]).reshape(-1,1)

def max_lik_estimate(X, y):
    theta_ml = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta_ml

def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta
    return prediction

theta_ml = max_lik_estimate(X, y)
print(theta_ml[0][0])

Xtest = np.linspace(-5, 5, 100).reshape(-1,1)
ml_prediction = predict_with_estimate(Xtest, theta_ml)

plt.figure()
plt.plot(X, y, '+', markersize = 10 , label = 'Data points')
plt.plot(Xtest, ml_prediction, label = f'y=[{theta_ml[0][0]}]x')
plt.xlabel ("X-axis ($x$)")
plt.ylabel ("y-axis ($y$)")
plt.title ("Plot of Training Data Set")
plt.xlim([-5,5])
plt.legend()
plt.grid(True)
plt.show()