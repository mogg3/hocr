import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import re
from tensorflow import keras
from tensorflow.keras.layers import *
import time
import category_encoders as ce
import pickle


def new_model(model_name, seed, img_am, epochs):
    model, chars = train_model(seed, img_am, epochs)
    model.save(f'training/models/{model_name}')
    return model, chars


def load_model(model_name):
    return keras.models.load_model(f"training/models/{model_name}")


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def load_img(src):
    image = cv.imread(src)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = image.reshape(32, 32, 1)
    return image


def big_small_split(folder_source):
    files = os.listdir(folder_source)
    small = []
    big = []

    for file in files:
        if file[0] == '_':
            small.append(file)
            if len(small) == 1300:
                break

    for file in files:
        if file[0] != '_':
            big.append(file)
            if len(big) == 700:
                break

    return {'small': small, 'big': big}


def get_dict(src, img_am):
    # split = ['A', 'B', 'E', 'F', 'G', 'H', 'Q', 'R']
    split = ['A', 'E', 'H']

    start = time.time()
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    ignore = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.DS_Store', 'Q', 'X']
    counter = 1300
    for i in range(img_am):
        if counter == 700:
            ignore.remove('Q')
            ignore.remove('X')
        if counter == -300:
            ignore.append('Q')
            ignore.append('X')
        for folder in os.listdir(src):
            if folder in ignore:
                continue
            if folder in split:
                if folder not in char_set:
                    char_set[folder] = big_small_split(f'{src}/{folder}')
                try:
                    img_small = load_img(f"{src}/{folder}/{char_set[folder]['small'][i]}")
                    char_dict["char"].append(folder.lower())
                    char_dict["matrix"].append(img_small)
                    img_big = load_img(f"{src}/{folder}/{char_set[folder]['big'][i]}")
                    char_dict["char"].append(folder)
                    char_dict["matrix"].append(img_big)
                except IndexError:
                    continue
            else:
                char_dict["char"].append(folder)
                if folder not in char_set:
                    char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
                img = load_img(f"{src}/{folder}/{char_set[folder][counter]}")
                char_dict["matrix"].append(img)
        counter -= 1

    stop = time.time()
    print(stop-start, 'seconds')
    return char_dict


def get_data(src, img_am):
    char_dict = get_dict(src, img_am)
    X = np.array(char_dict['matrix'])/255
    le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore", use_cat_names = True)
    y = np.array(char_dict['char'])
    y = le.fit_transform(y)
    feature_names = [value[2] for value in le.get_feature_names()]
    with open('training/feature_names.txt', 'wb') as file:
        pickle.dump(feature_names, file)
    split = int(round(0.9 * len(X)))

    train_X = X[:split]
    train_y = y[:split]
    test_X = X[split:]
    test_y = y[split:]

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    return train_X, train_y, test_X, test_y


def train_model(seed, img_am, epochs):

    train_X, train_y, test_X, test_y = get_data(r"../data/training_data", img_am)
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
    model.add(Dense(29, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Fitting...')
    model.fit(train_X, train_y, epochs=epochs)
    print('Done fitting...')
    results = model.evaluate(test_X, test_y)
    print(results)

    return model

