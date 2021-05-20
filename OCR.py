from text_segmentation import img_segmentation, img_to_input
import cv2 as cv
import os
from training.train_model import train_model, load_model, new_model, load_img, get_dict
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np


def reverse_one_hot(pred):
    with open('training/feature_names.txt', 'rb') as file:
        feature_names = pickle.load(file)
    return feature_names[pred.index(max(pred))]


def ocr(img_src):
    # model = load_model('split_AEH_less_QX_700big_1300small_all_separated_model')
    input_images = img_segmentation(img_src)
    result_string = ""
    for image in input_images:
        if image == ' ':
            result_string += image
        else:
            cv.imwrite("tmp.jpg", image)
            input_arr = load_img('tmp.jpg')/255
            input_arr = np.reshape(input_arr, (1, 32, 32, 1))
            pred = list(model.predict(input_arr)[0])
            os.remove('tmp.jpg')
            result_string += reverse_one_hot(pred)
    print(result_string)


for s in range(4, 9):
    model = new_model(f'split_AEH+{s}_less_QX_700big_1300small_all_separated_model', 23, 2000, 25, s)



