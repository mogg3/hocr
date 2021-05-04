import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from tensorflow import keras
import time
import pickle
import category_encoders as ce


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def load_img(src):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not img.shape == (32, 32):
        img = cv2.resize(img, (32, 32))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_not(img)
    img = img.flatten()
    img = img.astype('int32')
    return img


def train_neural():
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    src = r"model/dataset/handwritten_letters"
    print('Loading images...')
    for i in range(1000):
        for folder in os.listdir(src):
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
            img = load_img(f"{src}/{folder}/{char_set[folder][i]}")
            char_dict["matrix"].append(img)
            char_dict["char"].append(folder)

    le = ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
    y = np.array(char_dict['char'])
    y = le.fit_transform(y)

    X = np.array(char_dict["matrix"])

    split = int(round(0.9 * len(X)))

    train_X = X[:split] / 255
    train_y = y[:split]
    test_X = X[split:] / 255
    test_y = y[split:]

    train_X = train_X.astype('int32')
    test_X = test_X.astype('int32')

    train_X = tf.constant(train_X)
    train_y = tf.constant(train_y)

    test_X = tf.constant(test_X)
    test_y = tf.constant(test_y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(500, activation='relu', input_dim=1024))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(35, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Training...')
    model.fit(train_X, train_y, epochs=20)
    #pred_train = model.predict(train_X)
    print('Done')
    scores = model.evaluate(train_X, train_y, verbose=0)
    print('-'*30, 'Model stats', '-'*30)
    print(f'Accuracy on training data: {scores[1]}% \n Error on test data: {1 - scores[1]}')
    print()
    #pred_test = model.predict(test_X)
    scores2 = model.evaluate(test_X, test_y, verbose=0)
    print(f'Accuracy on test data: {scores2[1]}% \n Error on test data: {1 - scores2[1]}')
    print('-' * 73)
    return model


def get_wrong_predictions(predictions):
    in_test_y = char_dict['char'][split:]
    chars = char_dict['char'][:35]
    print("{Actual letter} - {Predicted letter}")
    for i, prediction in enumerate(predictions):
        prediction = list(prediction)
        actual_letter = in_test_y[i]
        predicted_letter = chars[prediction.index(max(prediction))]
        if predicted_letter != actual_letter:
            print(actual_letter, "-", predicted_letter)


def predict_single_img(model, img, chars):
    prediction = model.predict(np.array([img, ])/255)
    prediction = list(prediction[0])
    return chars[prediction.index(max(prediction))]
