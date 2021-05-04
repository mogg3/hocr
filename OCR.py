from text_segmentation import img_segmentation, image_paste
import cv2 as cv
import os
from nn_model import train_neural, predict_single_img, load_img, sorted_alphanumeric
import pickle
from tensorflow import keras


def new_model(model_typ, file_name):
    model = None
    if file_name in os.listdir("model"):
        os.remove(f"model/{file_name}")
    if model_typ.lower() == "neural":
        model = train_neural()
        model.save('model/{file_name}')
    if model_typ.lower() == "forest":
        model = train_forest()
        with open(f"model/{file_name}", "wb") as file:
            pickle.dump(model, file)
    return model


def load_model(file_name):
    with open(f"model/{file_name}", "rb") as file:
        model = pickle.load(file)
    return model


def random_forest_ocr():
    model = load_model(file_name)
    # model = new_model(model_typ="forest")

    cropped_images = img_segmentation('input.tif')

    for image in cropped_images:
        print(model.predict([image.flatten()]))
        image = cv.resize(image, (100, 100))
        cv.imshow('image', image)
        cv.waitKey(0)


def neural_network_ocr():
    # New model
    # model = new_model('neural', 'nn_model')

    # Load model
    # model = keras.models.load_model('model/nn_model')

    cropped_images = img_segmentation('input.tif')

    # TODO Fix ugly reversed onehot encoding
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    src = r"model/dataset/handwritten_letters"
    print('Loading images...')
    for i in range(1000):
        for folder in os.listdir(src):
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
            char_dict["matrix"].append('-')
            char_dict["char"].append(folder)

    chars = char_dict['char'][:35]

    for image in cropped_images:
        print(predict_single_img(model, image.flatten(), chars))
        image = cv.resize(image, (100, 100))
        cv.imshow('image', image)
        cv.waitKey(0)
