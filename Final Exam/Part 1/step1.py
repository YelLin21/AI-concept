import numpy as np
import matplotlib.pyplot as plt

#step 1
np.random.seed(42)
N=10

X0 = np.random.randn(N,2) * 0.5 + np.array([1,0])
y0 = np.zeros(N, dtype=int)

X1= np.random.randn(N,2) * 0.5 + np.array([3,3])
y1= np.ones(N, dtype=int)

X2= np.random.randn(N,2) * 0.5 + np.array([5,5])
y2= np.full(N,2, dtype=int)

plt.figure()
plt.plot(X0, color='blue', markersize =10, label = 'Bronze')
plt.plot(X1, color='red', markersize =10, label = 'Silver')
plt.plot(X2,color='green', markersize =10, label = 'Gold')
plt.xlabel ("Member Age")
plt.ylabel ("MOnthly Purchases")
plt.legend()
plt.show()

#step2
X= np.vstack((X0,X1,X2))
y= np.concatenate((y0,y1,y2))

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X=X[indices]
y=y[indices]

split_idx= int(0.8 *len(X))
X_train,X_test = X[:split_idx], X[split_idx:]
y_train,y_test = y[:split_idx], y[split_idx:]

num_classes = 3

def to_one_hot(labels, num_classes):
    return np.eye (num_classes) [labels]

y_train = to_one_hot (y_train , num_classes)
y_test = to_one_hot (y_test, num_classes)

def init_parameters (input_dim =2 , hidden_dim = 4, output_dim = 3):
    np.random.seed(42)
    W1 = 0.01 * np.random.randn (input_dim , hidden_dim)
    b1 = np.zeros ((1,hidden_dim))
    W2 = 0.01 * np.random.randn (hidden_dim, output_dim)
    b2 = np.zeros ((1,output_dim))
    return W1,b1, W2, b2

W1,b1, W2, b2 = init_parameters(input_dim= 2, hidden_dim= 4, output_dim=3)

print("The first two rows of X_train", X_train[[[0,0],[0,1]],
                                               [[1,0],[1,1]]])
print("The first two rows of X_test", X_test[[[0,0],[0,1]],
                                               [[1,0],[1,1]]])
print("Weights W1", W1)
print("Weights W2", W2)

#step3
def relu(z):
    return np.maximum(0,z)

def relu_deriv(z):
    return (z>0).astype (float)

def softmax(z):
    shifted = z - np.max (z, axis = 1, keepdims= True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims = True)

def forward_pass (X, W1, b1, W2, b2):
    Z1 = X.dot(W1) +b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    probs = softmax (Z2)
    return Z1, A1, Z2, probs

def cross_entropy_loss(probs, targets):
    eps = 1e-12
    return np.mean(np.sum(targets * np.log(probs + eps), axis= 1))

Z1, A1,Z2,probs = forward_pass (X_train[:2], W1, b1, W2,b2)
loss = cross_entropy_loss(probs, y_train[:2])

print("Z1 value: ", Z1)
print("A1 value: ", A1)
print("Z2 value: ", Z2)
print("probs value: ", probs)
print("Loss: ", loss)


# #step 4.2
def backward_pass(X, Y, Z1,A1,Z2, probs, W1, W2):
    N= X.shape[0]
    dZ2 = (probs -Y )/N
    dW2 = 
    db2 =

    dA1 =
    dZ1 =

    dW1 = 
    db1 = 

    return dW1, db1, dW2, db2


#step5
learning_rate= 0.01
def train(X_train, y_train, W1,b1,W2, b2, learning_rate = 0.01, epochs =1000):
    loss_history=[]
    for epoch in range (epochs):
        Z1,A1,Z2, probs= forward_pass (X_train, W1,b1,W2,b2)


        dW1,db1_,dW2,db2_ = backward_pass (X_train,)

#step 6.1

def predict (X, W1, b1, W2, b2):
    _, A1, _, probs = forward_pass(X, W1, b1, W2,b2)
    return np.argmax (probs, axis=1)

def accuracy (y_pred, y_true):
    return np.mean (y_pred == y_true)

y_pred_test = 
acc_test =
print (f"Test Accuracy : {acc_test *100:.2f}%")
