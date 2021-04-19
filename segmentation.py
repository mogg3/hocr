import cv2 as cv
import numpy as np

image = cv.imread('input.tif')

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
    boxes.append((x, y, w, h))
    rectangle = cv.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv.waitKey(0)

cv.imshow('marked areas',image)
cv.waitKey(0)
