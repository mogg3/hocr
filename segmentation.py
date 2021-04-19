import cv2 as cv
import numpy as np


class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.top_left = (x, y)
        self.top_right = (x + width, y)
        self.bottom_left = (x, y + height)
        self.bottom_right = (x + width, y + height)
        self.area = width * height


def overlap(R1, R2):
    if (R1.x >= (R2.width + R2.x)) or ((R1.width + R1.x) <= R2.x) or ((R1.y + R1.height) <= R2.y) or (R1.y >= (R2.y + R2.height)):
        return False
    else:
        return True


def get_overlaps(box, boxes):
    overlaps = []
    for i in range(len(boxes)):
        if overlap(box, boxes[i]) and box != boxes[i]:
            overlaps.append(boxes[i])
    return overlaps


image = cv.imread('input2.tif')

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
ret, thresh = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 6), (1, 1))
eroded = cv.erode(thresh, kernel, iterations=1)

#cv.imshow('image', eroded)
#cv.waitKey(0)

ctrs, hier = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])

boxes = []

for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv.boundingRect(ctr)
    boxes.append(Rectangle(x, y, w, h))
    cv.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

cv.imshow('marked areas',image)
cv.waitKey(0)

dicts = []
for i, box in enumerate(boxes):
    dict_ = {i: get_overlaps(box, boxes)}
    dicts.append(dict_)

print(dicts)



