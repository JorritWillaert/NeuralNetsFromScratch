import numpy as np

def backward_propagation(m, A, X, Y):
    dw = 1 / m * (np.dot(X, ((A - Y).T)))
    db = 1 / m * (np.sum(A - Y))
    return dw, db
