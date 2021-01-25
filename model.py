import numpy as np
import matplotlib.pyplot as plt
import cv2

import helpers as hp

def model():
    """Define model"""
    width = 250
    height = 250
    hp.preprocess_all_data(width, height)
    X, y, m, n = hp.create_X_and_y(width, height, train = True)

    #Simple logistic regression network.
    # X = (num_px * num_px * 3, number_of_examples)

if __name__ == "__main__":
    model()
