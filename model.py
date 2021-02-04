import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import tensorflow
from tensorflow.keras.datasets import mnist

import helpers as hp
import forward_propagation as fp
import backward_propagation as bp
import testing as tst

np.set_printoptions(threshold=sys.maxsize)

def model():
    """Define model"""
    width = 28
    height = 28
    num_iterations = 20
    learning_rate = 0.1
    """
    hp.preprocess_all_data(width, height)
    X, Y, m, n = hp.create_X_and_y(width, height, suffix = "train", train = True)
    """
    (X, Y), (X_test, Y_test) = mnist.load_data()
    # Flatten the images
    image_vector_size = 28*28
    m = X.shape[0]
    m_test = X_test.shape[0]
    X = X.reshape(m, image_vector_size)
    Y = Y.reshape(m, 1)
    print(X.shape)
    X_test = X_test.reshape(m_test, image_vector_size)
    n = X.shape[1]
    w, b = hp.initialize_parameters(width * height, output = 1) #Removed the * 3 because only black/white images in MNIST
    print(w.shape)
    print(Y.shape)
    costs = []
    for i in range(num_iterations):
        A = fp.forward_propagation(w, b, X)
        cost = hp.calculate_cost(m, A, Y)
        print(cost)
        costs.append(cost)
        dw, db = bp.backward_propagation(m, A, X, Y)
        print(b, "Before")
        w, b = hp.update_parameters(w, b, dw, db, learning_rate)
        print(b, "After")
    #X_test, Y_test, m_test, n = hp.create_X_and_y(width, height, suffix = "train", train = False)
    accuracy_training = tst.test_accuracy(X, Y, m, w, b)
    accuracy_test = tst.test_accuracy(X_test, Y_test, m_test, w, b)
    print("The accuracy of the training set equals {:0.2f} %".format(accuracy_training * 100))
    print("The accuracy of the test set equals {:0.2f} %".format(accuracy_test * 100))
    plt.plot(costs)
    plt.xlabel('# iterations')
    plt.ylabel('Cost')
    plt.show()

if __name__ == "__main__":
    model()
