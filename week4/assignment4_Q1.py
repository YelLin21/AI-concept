import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#step1
X= np.array([-4.5, -3.5, -3, -1.8, -0.2, 0.3, 1.3, 2.6, 3.8, 4.8]).reshape(-1,1) # 10x1 vector, N=5, D=
y = np.array([
    [-0.91650116],
    [-0.47546053], 
    [-0.10972425],
    [ 0.29504095],
    [-0.01596218],
    [ 0.10014049], 
    [0.48108203],
    [0.10979023], 
    [-0.99742128],
    [-0.91271826]
]).reshape(-1,1) # 10x1 vector

X_test = np.array([-3.99, -1.38, -1.37,-0.94,0.69, 1.4, 1.57, 1.78, 1.81, 4.89]).reshape(-1,1) # 10x1 vector, N=5, D=1
y_test = np.array([
    [-0.80737607],
    [0.19813376], 
    [0.19537639], 
    [0.07185977],
    [0.24954213], 
    [0.50662504], 
    [0.52943298],
    [0.52406997], 
    [0.51999057], 
    [-0.82318208]
]).reshape(-1,1) # 10x1 vector

#step2
def poly_features(X,K):
    X= X.flatten()
    N= X.shape[0]
    Phi= np.zeros((N,K+1))
    for k in range (K+1):
        Phi[:,k]= X**k
    return Phi

#step3
def nonlinear_features_maximum_likelihood(Phi, y):

    theta_ml = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    return theta_ml

K=4
Phi = poly_features(X,K)

theta_ml = nonlinear_features_maximum_likelihood(Phi, y)
print (theta_ml)

#step4
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Compute RMSE for Training and Test Datasets
degrees = range(16)
rmse_train = []
rmse_test = []

for d in degrees:
    # Train Data
    Phi_train = poly_features(X, d)
    theta = nonlinear_features_maximum_likelihood(Phi_train, y)
    y_pred_train = Phi_train @ theta
    rmse_train.append(rmse(y, y_pred_train))
    
    # Test Data
    Phi_test = poly_features(X_test, d)
    y_pred_test = Phi_test @ theta
    rmse_test.append(rmse(y_test, y_pred_test))

print(f"Training RMSE: {rmse_train}")
print(f"Test RMSE: {rmse_test}")

#step5

optimal_k = degrees[np.argmin(rmse_test)]
print(f"The optimal polynomial degree is: {optimal_k}")

X_test_all = np.linspace(-5, 5, 100).reshape(-1, 1)
Phi_test_all = poly_features(X_test_all, optimal_k)
theta_optimal = nonlinear_features_maximum_likelihood(poly_features(X, optimal_k), y)
ypred_test_all = Phi_test_all @ theta_optimal

plt.scatter(X, y, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_test_all, ypred_test_all, color='green', label=f'Polynomial Degree {optimal_k}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Optimal Polynomial Model Fit')
plt.legend()
plt.grid(True)
plt.show()