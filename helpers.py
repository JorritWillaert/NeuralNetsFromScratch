import cv2
import os

def preprocessing(img_name, train_test):
    img = cv2.imread('DataSetCatsDogs/' + train_test + '/' + train_test + '/' + img_name)
    desired_width = 250
    desired_height = 250
    dim = (desired_width, desired_height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def preprocess_all_data():
    for train_test in ['train', 'test1']:
        if len(os.listdir('DataSetCatsDogs/' + train_test + '/preprocessed_' + train_test)) == 0:
            dir_list = sorted(os.listdir('DataSetCatsDogs/' + train_test + '/' + train_test), key=len)
            for img_num in range(len(dir_list)):
                img_name = dir_list[img_num]
                resized_img = preprocessing(img_name, train_test)
                cv2.imwrite('DataSetCatsDogs/' + train_test + '/preprocessed_' + train_test + '/' + img_name, resized_img)
