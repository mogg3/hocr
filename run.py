from text_segmentation import img_segmentation
from image_paste import image_paste
import cv2 as cv

img_path = 'input3.tif'
cropped_images = img_segmentation(img_path)
pasted_images = image_paste(cropped_images)

cv.imshow('Origin', cv.imread(img_path))
cv.waitKey(0)

for image in pasted_images:
    cv.imshow('result', image)
    cv.waitKey(0)
