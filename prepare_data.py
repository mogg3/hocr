import os
import cv2 as cv


class Character:
    def __init__(self, vector, character):
        self.vector = vector
        self.character = character


def to_matrix(file_path):
    img = cv.imread(file_path)
    edges = cv.Canny(img, 100, 200)
    return edges


def get_dicts(path, limit, type):
    dicts = []
    for char_folder in os.listdir(f"{path}/{type}"):
        file_path = f"{path}/{type}/{char_folder}"
        for i, file in enumerate(os.listdir(file_path)[0:limit]):
            img = cv.imread(f"datasets/handwritten_letters/{type}/{char_folder}/{file}")
            edges = cv.Canny(img, 100, 200)
            dict = {'matrix': edges, 'character': char_folder}
            dicts.append(dict)
            #print(f"{char_folder} - {i+1}/{len(os.listdir(file_path))}")
    return dicts


def vectorize(matrix):
    vector = []
    for row in matrix:
        for element in row:
            vector.append(element)
    return vector


def get_data(dicts):
    data = []
    for dict in dicts:
        vector = vectorize(dict['matrix'])
        for i, element in enumerate(vector):
            if element == 255:
                vector[i] = 1
        char = Character(vector, dict['character'])
        data.append(char)
    return data


def to_ones(vectors):
    for vector in vectors:
        for i, element in enumerate(vector):
            if element == 255:
                vector[i] = 1
    return vectors


file_path = 'datasets/handwritten_letters/'
limit = 10

training_dicts = get_dicts(file_path, limit, 'Train')
validation_dicts = get_dicts(file_path, limit, 'Validation')

training_data = get_data(training_dicts)
validation_data = get_data(validation_dicts)

# Character(vector (x), character (y))

# Bokstav med minst bilder 4261, J
# Bokstav med max bilder 65503, 0
