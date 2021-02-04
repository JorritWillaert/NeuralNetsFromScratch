import numpy as np

def backward_propagation(m, A, X, Y):
    dw = 1 / m * (np.dot(X.T, ((A - Y))))
    db = 1 / m * (np.sum(A - Y))
    #print(dw, db, "gradients")
    return dw, db
