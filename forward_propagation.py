import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagation(w, b, X):
    Z = np.dot(X, w) + b
    A = sigmoid(Z)
    #print("Shape A", A.shape)
    return A
