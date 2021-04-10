import os
import cv2 as cv


class Character:
    def __init__(self, vector, character):
        self.vector = vector
        self.character = character


def get_matrix(file_path):
    img = cv.imread(file_path)
    edges = cv.Canny(img, 100, 200)
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges


def get_dicts(file_path):
    matrixes = []
    for char_folder in os.listdir(file_path):
        for file in os.listdir(f"{file_path}/{char_folder}"):
            img = cv.imread(f"datasets/handwritten_letters/Train/{char_folder}/{file}")
            edges = cv.Canny(img, 100, 200)
            dict = {'matrix': edges, 'character': char_folder}
            matrixes.append(dict)
        break
    return matrixes


def vectorize(matrix):
    vector = []
    for row in matrix:
        for element in row:
            vector.append(element)
    return vector


def get_vectors(matrixes):
    vectors = []
    for matrix in matrixes:
        vector = vectorize(matrix)
        for i, element in enumerate(vector):
            if element == 255:
                vector[i] = 1
        vectors.append(vector)
    return vectors


def to_ones(vectors):
    for vector in vectors:
        for i, element in enumerate(vector):
            if element == 255:
                vector[i] = 1
    return vectors


file_path = 'datasets/handwritten_letters/Train'

dicts = get_dicts(file_path)
#vectors = get_vectors(matrixes)

# x = bildens vector
# y = vilken bokstav som är skriven
