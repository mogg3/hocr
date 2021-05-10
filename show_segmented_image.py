from text_segmentation import img_segmentation, image_paste
import cv2 as cv


def get_small_images(img_path):
    return img_segmentation(img_path)


def show_images(images):
    for image in images:
        image = cv.resize(image, (100, 100))
        cv.imshow('image', image)
        cv.waitKey(0)


images = get_small_images('input.tif')
show_images(images)
