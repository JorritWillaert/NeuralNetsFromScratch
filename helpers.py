import cv2

def preprocessing(cat_or_dog, img_number):
    img = cv2.imread('DataSetCatsDogs/train/train/' + cat_or_dog + '.' + str(img_number) + '.jpg')
    desired_width = 250
    desired_height = 250
    dim = (desired_width, desired_height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img
