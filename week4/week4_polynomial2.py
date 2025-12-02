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

def poly_features(X,K):
    X= X.flatten()
    N= X.shape[0]
    Phi= np.zeros((N,K+1))
    for k in range (K+1):
        Phi[:,k]= X**k
    return Phi

def nonlinear_features_maximum_likelihood(Phi, y):

    kappa = 1e-08 #'jitter' term; fod for numerical stability
    D = Phi.shape[1]

    Pt = Phi.T@y
    PP= Phi.T @ Phi + kappa*np.eye(D)

    C=scipy.linalg.cho_factor(PP)
    theta_ml = scipy.linalg.cho_solve(C,Pt)
    return theta_ml

K=10
Phi = poly_features(X,K)

theta_ml = nonlinear_features_maximum_likelihood(Phi, y)

Xtest = np.linspace(-6, 6, 100).reshape(-1,1)

Phi_test = poly_features(Xtest,K)

y_pred = Phi_test @ theta_ml
print(theta_ml[0][0])
plt.figure()
plt.plot(X, y, '+', markersize = 10 , label = 'Data points')
plt.plot(Xtest, y_pred, label=f'y = {theta_ml[0][0]}x')
plt.xlabel ("X-axis ($x$)")
plt.ylabel ("y-axis ($y$)")
plt.title ("Plot of Training Data Set")
plt.xlim([-6,6])
plt.legend()
plt.grid(True)
plt.show()