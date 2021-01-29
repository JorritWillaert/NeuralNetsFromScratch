import numpy as np
import sys

import forward_propagation as fp
np.set_printoptions(threshold=sys.maxsize)

def test_accuracy(X_test, Y_test, m_test, w, b):
    A = fp.forward_propagation(w, b, X_test)
    sum = 0
    for i in range(m_test):
        if (A[0][i] > 0.5 and Y_test[0][i] == 1) or (A[0][i] < 0.5 and Y_test[0][i] == 0):
                sum += 1
    accuracy = sum / m_test
    return accuracy
