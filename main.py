import os
import cv2 as cv


def get_matrix(file_path):
    img = cv.imread(file_path)
    edges = cv.Canny(img, 100, 200)
    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    return edges


def get_matrixes(file_list):
    matrixes = []
    for file in file_list:
        img = cv.imread(f"datasets/handwritten_letters/Train/0/{file}")
        edges = cv.Canny(img, 100, 200)
        matrixes.append(edges)
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
        vectors.append(vector)
    return vectors


def to_ones(vectors):
    for vector in vectors:
        for i, element in enumerate(vector):
            if element == 255:
                vector[i] = 1
    return vectors


file_path = 'datasets/handwritten_letters/Train/0/0.jpg'
file_list = os.listdir('datasets/handwritten_letters/Train/A/')[0:10]

matrixes = get_matrixes(file_list)
vectors_ = get_vectors(matrixes)
vectors = to_ones(vectors_)
