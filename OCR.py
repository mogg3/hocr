from text_segmentation import img_segmentation, image_paste
import cv2 as cv
import os
from training.train_model import train_model, load_model, new_model, load_img, get_data, sorted_alphanumeric
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np


def get_chars():
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    ignore = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '.DS_Store']
    src = r'data/training_data'
    for i in range(26):
        for folder in os.listdir(src):
            if folder in ignore:
                continue
            char_dict["char"].append(folder)
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
    return char_dict['char'][:26]


def get_chars_digits():
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    src = r'data/training_data'
    for i in range(35):
        for folder in os.listdir(src):
            if folder == '.DS_Store':
                continue
            char_dict["char"].append(folder)
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
    return char_dict['char'][:35]


def ocr():
    # model = new_model(23, 1000, 25)
    no_digits_model = load_model('training/no_digits_model')
    model = load_model('training/model')

    images = img_segmentation('input.tif')
    chars_digits = get_chars_digits()
    chars = get_chars()

    for image in images:
        cv.imwrite("tmp.jpg", image)
        image = tf.keras.preprocessing.image.load_img('tmp.jpg', grayscale=True, color_mode="grayscale")
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array(input_arr)/255
        input_arr = np.reshape(input_arr, (1, 32, 32, 1))

        pred = list(model.predict(input_arr)[0])
        print('With digits prediction: ', chars_digits[pred.index(max(pred))])

        no_digits_pred = list(no_digits_model.predict(input_arr)[0])
        print('No digits prediction: ', chars[no_digits_pred.index(max(no_digits_pred))])

        image = cv.imread('tmp.jpg')
        image = cv.resize(image, (100, 100))
        os.remove('tmp.jpg')
        cv.imshow('image', image)
        cv.waitKey(0)
        print('-'*10)
