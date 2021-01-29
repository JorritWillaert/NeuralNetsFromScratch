import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

import helpers as hp
import forward_propagation as fp
import backward_propagation as bp
import testing as tst

np.set_printoptions(threshold=sys.maxsize)

def model():
    """Define model"""
    width = 64
    height = 64
    num_iterations = 250
    learning_rate = 0.002
    hp.preprocess_all_data(width, height)
    X, Y, m, n = hp.create_X_and_y(width, height, suffix = "train", train = True)
    w, b = hp.initialize_parameters(width * height * 3, output = 1) #Logistic regression: only one output layer now
    costs = []
    for i in range(num_iterations):
        A = fp.forward_propagation(w, b, X)
        cost = hp.calculate_cost(m, A, Y)
        print(cost)
        costs.append(cost)
        dw, db = bp.backward_propagation(m, A, X, Y)
        w, b = hp.update_parameters(w, b, dw, db, learning_rate)
    X_test, Y_test, m_test, n = hp.create_X_and_y(width, height, suffix = "train", train = False)

    accuracy_training = tst.test_accuracy(X, Y, m, w, b)
    accuracy_test = tst.test_accuracy(X_test, Y_test, m_test, w, b)
    print("The accuracy of the training set equals {:0.2f} %".format(accuracy_training * 100))
    print("The accuracy of the test set equals {:0.2f} %".format(accuracy_test * 100))

if __name__ == "__main__":
    model()
