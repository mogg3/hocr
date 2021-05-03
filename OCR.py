from text_segmentation import img_segmentation, image_paste
from model import new_model, load_model
import cv2 as cv

# model = load_model()
model = new_model(model_typ="forest")

cropped_images = img_segmentation('input3.tif')

for image in cropped_images:
    print(model.predict([image.flatten()]))
    image = cv.resize(image, (100, 100))
    cv.imshow('image', image)
    cv.waitKey(0)

