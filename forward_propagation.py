import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def forward_propagation(w, b, X):
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    return A
