import numpy as np
import matplotlib.pyplot as plt
import cv2

import helpers as hp
import forward_propagation as fp

def model():
    """Define model"""
    width = 250
    height = 250
    num_iterations = 100
    hp.preprocess_all_data(width, height)
    X, Y, m, n = hp.create_X_and_y(width, height, train = True)
    w, b = hp.initialize_parameters(width * height * 3, output = 1) #Logistic regression: only one output layer now
    costs = []
    for i in range(num_iterations):
        A = fp.forward_propagation(w, b, X)
        cost = hp.calculate_cost(m, A, Y)
        print(cost)
        costs.append(cost)


if __name__ == "__main__":
    model()
