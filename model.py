import numpy as np
import matplotlib.pyplot as plt
import cv2

import helpers as hp
import forward_propagation as fp
import backward_propagation as bp

def model():
    """Define model"""
    width = 250
    height = 250
    num_iterations = 100
    learning_rate = 0.1
    path_ssd_drive = "C:/DatasetCatsDogs/"
    hp.preprocess_all_data(width, height)
    X, Y, m, n = hp.create_X_and_y(width, height, path_ssd_drive, train = True)
    w, b = hp.initialize_parameters(width * height * 3, output = 1) #Logistic regression: only one output layer now
    costs = []
    for i in range(num_iterations):
        A = fp.forward_propagation(w, b, X)
        cost = hp.calculate_cost(m, A, Y)
        print(cost)
        costs.append(cost)
        dw, db = bp.backward_propagation(m, A, X, Y)
        w, b = hp.update_parameters(w, b, dw, db, learning_rate)

if __name__ == "__main__":
    model()
