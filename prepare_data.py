import os
import cv2 as cv


class Character:
    def __init__(self, vector, character):
        self.vector = vector
        self.character = character


def to_matrix(file_path):
    image = cv.imread(file_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    gray = cv.bitwise_not(gray)
    return gray


def get_dicts(path, limit):
    dicts = []
    for char_folder in os.listdir(f"{path}"):
        folder_path = f"{path}/{char_folder}"
        for i, file in enumerate(os.listdir(folder_path)[0:limit]):
            matrix = to_matrix(f"datasets/handwritten_letters/{char_folder}/{file}")
            char_dict = {'matrix': matrix, 'character': char_folder}
            dicts.append(char_dict)
            #print(f"{char_folder} - {i+1}/{len(os.listdir(file_path))}")
    return dicts


def vectorize(matrix):
    vector = []
    for row in matrix:
        for element in row:
            vector.append(element)
    return vector


def get_data(src, limit):
    dicts = get_dicts(src, limit)
    data = []
    for dict in dicts:
        vector = vectorize(dict['matrix'])
        char = Character(vector, dict['character'])
        data.append(char)
    return data


src = 'datasets/handwritten_letters'
limit = 10
data = get_data(src, 10)


# Character(vector (x), character (y))
# Bokstav med minst bilder 4261, J
# Bokstav med max bilder 65503, 0
