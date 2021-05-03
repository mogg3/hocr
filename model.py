import cv2
import re
import os
import pickle
from sklearn.ensemble import RandomForestClassifier


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def new_model(model_typ="forest", file_name="model.dat"):
    model = None
    if file_name in os.listdir("model"):
        os.remove(f"model/{file_name}")
    if model_typ.lower() == "forest":
        model = train_forest()

    with open(f"model/{file_name}", "wb") as file:
        pickle.dump(model, file)
    return model


def load_model(file_name="model.dat"):
    with open(f"model/{file_name}", "rb") as file:
        model = pickle.load(file)
    return model


def train_forest():
    char_dict = {"char": [], "matrix": []}
    char_set = dict()
    ignore = ["#", "$", "&", "@"]
    src = r"model/dataset"
    for i in range(1000):
        for folder in os.listdir(src):
            if folder in ignore:
                continue
            if folder not in char_set:
                char_set[folder] = sorted_alphanumeric(os.listdir(src + '/' + folder))
            img = cv2.imread(f"{src}/{folder}/{char_set[folder][i]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if not img.shape == (32, 32):
                img = cv2.resize(img, (32, 32))
            img = img.flatten()
            char_dict["matrix"].append(img)
            char_dict["char"].append(folder)

    X = char_dict["matrix"]
    y = char_dict["char"]

    split = int(len(X) * 0.9)
    train_X = X[:split]
    train_y = y[:split]
    test_X = X[split:]
    test_y = y[split:]

    clf_rf = RandomForestClassifier(n_estimators=230, random_state=90)
    clf_rf.fit(train_X, train_y)
    y_pred = clf_rf.predict(test_X)
    # score = accuracy_score(test_y, y_pred)
    return clf_rf
