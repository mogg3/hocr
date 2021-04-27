import cv2 as cv
import numpy as np
from text_segmentation import img_segmentation


def image_paste(cropped_images):
    edges_list = []
    for cropped_image in cropped_images:
        blank_image = np.zeros((32, 32, 3), np.uint8)
        blank_image.fill(255)
        in_width = cropped_image.shape[1]
        in_height = cropped_image.shape[0]

        if in_width > in_height:
            scale_percent = 28 / in_width
            width = int(cropped_image.shape[1] * scale_percent)
            height = int(cropped_image.shape[0] * scale_percent)
            dim = (width, height)
            cropped_image = cv.resize(cropped_image, dim, interpolation=cv.INTER_AREA)
            x_offset = 2
            y_offset = int((32-height)/2)
            blank_image[y_offset:y_offset + cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = cropped_image

        elif in_width < in_height:
            scale_percent = 28 / in_height
            width = int(cropped_image.shape[1] * scale_percent)
            height = int(cropped_image.shape[0] * scale_percent)
            dim = (width, height)
            cropped_image = cv.resize(cropped_image, dim, interpolation=cv.INTER_AREA)
            x_offset = int((32 - width) / 2)
            y_offset = 2
            blank_image[y_offset:y_offset + cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = cropped_image

        edges = cv.Canny(blank_image, 100, 200)
        edges_list.append(edges)
    return edges_list

