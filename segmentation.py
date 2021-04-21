import cv2 as cv
import numpy as np
import os


class Box:
    def __init__(self, x, y, width, height, color, thickness):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.top_left = (x, y)
        self.top_right = (x + width, y)
        self.bottom_left = (x, y + height)
        self.bottom_right = (x + width, y + height)
        self.area = width * height
        self.color = color
        self.thickness = thickness


def process_img(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, processed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    return processed


def get_boxes(processed):
    ctrs, hier = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])
    boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append(Box(x, y, w, h, (0, 0, 255), 1))
    boxes = check_erode(boxes, processed)
    return boxes


def clean_boxes(boxes):
    return [box for box in boxes if box.height > 5 and box.width > 5]


def erode(boxes, processed):
    print("Eroding")
    boxes.clear()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 6), (1, 1))
    eroded = cv.erode(processed, kernel, iterations=1)
    ctrs, hier = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append(Box(x, y, w, h, (0, 0, 255), 1))
    return boxes


def check_erode(boxes, processed):
    counter = 0
    for box in boxes:
        if box.width > 80:
            counter += 1
    if counter >= 3:
        boxes = erode(boxes, processed)
    return boxes


def check_overlap(box1, box2):
    if (box1.x >= (box2.width + box2.x)) or ((box1.width + box1.x) <= box2.x) or ((box1.y + box1.height) <= box2.y) or (box1.y >= (box2.y + box2.height)):
        return False
    else:
        return True


def get_overlapping_boxes(box, boxes):
    overlaps = []
    for i in range(len(boxes)):
        if check_overlap(box, boxes[i]) and box != boxes[i]:
            overlaps.append(boxes[i])
    return overlaps


def get_dicts(boxes):
    return {box: get_overlapping_boxes(box, boxes) for box in boxes if get_overlapping_boxes(box, boxes)}


def inside(box1, box2):
    if (box1.x < box2.x) and (box1.y < box2.y) and (box1.x + box1.width > box2.x + box2.width) and (box1.y + box1.height > box2.y + box2.height):
        return True
    else:
        return False


def remove_inside_boxes(boxes):
    for box in boxes:
        for box_ in boxes:
            if inside(box, box_) and box != box_:
                boxes.remove(box_)
    return boxes


def divide_boxes(mean, boxes):
    new_boxes = []
    for i, box in enumerate(boxes):
        n = box.width // (mean * 0.7)
        if n >= 2:
            for j in range(int(n)):
                new_w = round((box.width / n))
                new_x = round(box.x + (new_w * (j)))
                new_y = box.y
                new_h = box.height
                new_box = Box(new_x, new_y, new_w, new_h, (255, 0, 0), 1)
                new_boxes.append(new_box)
    return boxes + new_boxes


def check_mean_width(boxes):
    box_width = sorted([box.width for box in boxes])
    mean = np.mean(box_width)
    if box_width[-1] >= mean*2.2:
        boxes = divide_boxes(mean, boxes)
    return boxes


def show_boxes(boxes, image):
    for box in boxes:
        cv.rectangle(image, (box.x, box.y), box.bottom_right, box.color, box.thickness)
    cv.imshow('marked areas', image)
    cv.waitKey(0)


def img_segmentation(img_path):
    image = cv.imread(img_path)
    processed = process_img(image)
    boxes = get_boxes(processed)
    boxes = clean_boxes(boxes)
    boxes = remove_inside_boxes(boxes)
    boxes = check_mean_width(boxes)
    show_boxes(boxes, image)


img_path = 'input.tif'

img_segmentation(img_path)
