import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from tensorflow import keras
from tensorflow.keras.layers import *
import time
import category_encoders as ce


def new_model(seed, img_am, epochs):
    # if 'model' in os.listdir("training"):
    #     os.remove("training/model")
    model = train_model(seed, img_am, epochs)
    model.save(f'training/no_digits_model')
    return model


def load_model(path):
    return keras.models.load_model(path)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def load_img(src):
    image = tf.keras.preprocessing.image.load_img(src, grayscale=True, color_mode = "grayscale")
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array(input_arr)
    return input_arr


def get_dict(src, img_am):
    start = time.time()
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    ignore = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '.DS_Store']
    for i in range(img_am):
        for folder in os.listdir(src):
            if folder in ignore:
                continue
            char_dict["char"].append(folder)
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
            img = load_img(f"{src}/{folder}/{char_set[folder][i]}")
            char_dict["matrix"].append(img)
    stop = time.time()
    print(stop-start, 'seconds')
    return char_dict


def get_data(src, img_am):
    char_dict = get_dict(src, img_am)
    X = np.array(char_dict['matrix'])/255
    le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
    y = np.array(char_dict['char'])

    y = le.fit_transform(y)

    split = int(round(0.9 * len(X)))

    train_X = X[:split]
    train_y = y[:split]
    test_X = X[split:]
    test_y = y[split:]

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    return train_X, train_y, test_X, test_y


def train_model(seed, img_am, epochs):
    train_X, train_y, test_X, test_y = get_data(r"data/training_data", img_am)
    tf.random.set_seed(seed)

    model = tf.keras.Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(26, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Fitting...')
    model.fit(train_X, train_y, epochs=epochs)
    print('Done fitting...')
    results = model.evaluate(test_X, test_y)
    print(results)

    return model
