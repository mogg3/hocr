from text_segmentation import img_segmentation, image_paste
import cv2 as cv
import os
from CNN_Model import train_cnn
import pickle
from tensorflow import keras


def new_model(model_typ, file_name, seed, img_am, epochs):
    model = None
    if file_name in os.listdir("model"):
        os.remove(f"model/{file_name}")
    if model_typ.lower() == "neural":
        model = train_cnn(seed, img_am, epochs)
        model.save(f'model/{file_name}')
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
    seed = 8
    img_am = 1000
    epochs = 20
    new_model('neural', 'nn_model', seed, img_am, epochs)

    # Load model
    # model = keras.models.load_model('model/nn_model')


neural_network_ocr()
