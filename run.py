from text_segmentation import img_segmentation, image_paste
import cv2 as cv


def get_small_images(img_path):
    cv.imshow('Origin', cv.imread(img_path))
    cv.waitKey(0)

    cropped_images = img_segmentation(img_path)
    pasted_images = image_paste(cropped_images)

    return pasted_images


def show_images(images):
    for image in images:
        image = cv.resize(image, (100, 100))
        cv.imshow('image', image)
        cv.waitKey(0)


path = 'input3.tif'
small_images = get_small_images(path)
show_images(small_images)
