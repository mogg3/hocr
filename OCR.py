from text_segmentation import img_segmentation, img_to_input
import cv2 as cv
import os
from training.train_model import train_model, load_model, new_model, load_img, get_dict
import pickle
from tensorflow import keras
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def reverse_one_hot(pred, feature_names_file):
    with open(f'training/feature_names/{feature_names_file}', 'rb') as file:
        feature_names = pickle.load(file)
    return feature_names[pred.index(max(pred))]


def ocr(img_name):
    model_name = 'split_AEH+6_less_QX_700big_1300small_all_separated_model'
    model = load_model(model_name)
    input_images = img_segmentation(f"test_images/{img_name}")
    result_string = ""
    for image in input_images:
        if type(image) == str:
            result_string += image
        else:
            cv.imwrite("tmp.jpg", image)
            input_arr = load_img('tmp.jpg')/255
            input_arr = np.reshape(input_arr, (1, 32, 32, 1))
            pred = list(model.predict(input_arr)[0])
            os.remove('tmp.jpg')
            feature_names_file = f'{model_name}.txt'
            result_string += reverse_one_hot(pred, feature_names_file).lower()
    return result_string


def presentation():
    print(f'Alphabet: {ocr("small_letters1.png")}{ocr("small_letters2.png")}')
    print(f'Handwritten text: {ocr("hej_jag_heter_marcus_redovisnings_bild.png")}')


def ocr_multiple_models(file_name):
    input_images = img_segmentation(f"test_images/{file_name}")
    for model_name in os.listdir('training/modelss'):
        model = load_model(model_name)
        result_string = ""
        for image in input_images:
            if image == ' ':
                result_string += image
            else:
                cv.imwrite("tmp.jpg", image)
                input_arr = load_img('tmp.jpg') / 255
                input_arr = np.reshape(input_arr, (1, 32, 32, 1))
                pred = list(model.predict(input_arr)[0])
                os.remove('tmp.jpg')
                feature_names_file = f'{model_name}.txt'
                result_string += reverse_one_hot(pred, feature_names_file).lower()
        print('Image: ', file_name)
        print('Result: ', result_string)
        print('Model: ', model_name)
        print('-'*20)

