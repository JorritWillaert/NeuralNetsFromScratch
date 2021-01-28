import cv2
import os
import numpy as np

def preprocessing(img_name, train_test, width, height):
    img = cv2.imread('DataSetCatsDogs/' + train_test + '/' + train_test + '/' + img_name)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def preprocess_all_data(width, height):
    for train_test in ['train', 'test1']:
        if len(os.listdir('DataSetCatsDogs/' + train_test + '/preprocessed_' + train_test)) == 0:
            dir_list = sorted(os.listdir('DataSetCatsDogs/' + train_test + '/' + train_test), key=len)
            for img_num in range(len(dir_list)):
                img_name = dir_list[img_num]
                resized_img = preprocessing(img_name, train_test, width, height)
                cv2.imwrite('DataSetCatsDogs/' + train_test + '/preprocessed_' + train_test + '/' + img_name, resized_img)

def create_X_and_y(width, height, train):
    if train:
        suffix = "train"
    else:
        suffix = "test1"
    dir_list = sorted(os.listdir('DataSetCatsDogs/' + suffix + '/preprocessed_' + suffix), key=len)
    X = np.zeros((width * height * 3, 2000))
    Y = np.zeros((1, 2000))
    m = len(dir_list)
    n = width * height * 3
    for img_num in range(len(dir_list)):
        num = img_num
        if (img_num % 12500 > 999): #Only use small set now
            continue
        if img_num >= 12500:
            num = img_num - 12500
        img_name = dir_list[img_num]
        img = cv2.imread('DataSetCatsDogs/' + suffix + '/preprocessed_' + suffix + '/' + img_name)
        img = img.flatten() / 255
        #img = np.expand_dims(img, axis = 1)
        X[:, num] = img
        if img_name[0:3] == "cat":
            Y[0, num] = 1
        else:
            Y[0, num] = 0
    return X, Y, m, n

def initialize_parameters(input, output):
    w = np.random.randn(input, output) * 0.01
    b = np.zeros((1, output))
    return w, b

def calculate_cost(m, A, Y):
    cost = -1 / m * (np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))
    return cost

def update_parameters(w, b, dw, db, learning_rate):
    dw = w - learning_rate * dw
    db = b - learning_rate * db
    return dw, db
