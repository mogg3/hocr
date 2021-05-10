from text_segmentation import img_segmentation, image_paste
import cv2 as cv
import os
from training.train_model import train_model, load_model, new_model, load_img, get_data, sorted_alphanumeric
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np


def ocr():
    # New model
    # new_model('neural', 'nn_model', 23, 1000, 25)

    # Load model
    model = load_model('training/model')
    images = img_segmentation('input.tif')

    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    src = r"data/training_data"
    for i in range(35):
        for folder in os.listdir(src):
            char_dict["char"].append(folder)
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))

    char_dict = char_dict['char'][:35]

    for image in images:
        cv.imwrite("tmp.jpg", image)
        image = tf.keras.preprocessing.image.load_img('tmp.jpg', grayscale=True, color_mode="grayscale")
        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.array(input_arr)
        input_arr = np.reshape(input_arr, (1, 32, 32, 1))
        pred = list(model.predict(input_arr)[0])
        print(char_dict[pred.index(max(pred))])
        image = cv.imread('tmp.jpg')
        cv.imshow('image', image)
        cv.waitKey(0)
        os.remove('tmp.jpg')


ocr()
