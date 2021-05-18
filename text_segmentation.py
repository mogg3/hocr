import cv2 as cv
import numpy as np
import os
import time
import copy


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
        self.left = x
        self.right = x + width
        self.top = y
        self.bottom = y + height
        self.area = width * height
        self.color = color
        self.thickness = thickness

    def __repr__(self):
        return f'x = {self.x} y = {self.y} width = {self.width} height = {self.height}'


def process_img(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, processed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    return processed


def get_boxes(processed):
    ctrs, hier = cv.findContours(processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda contour: cv.boundingRect(contour)[0])
    boxes = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append(Box(x, y, w, h, (0, 0, 255), 1))
    boxes = check_erode(boxes, processed)
    return boxes


def get_mean(boxes):
    box_width = sorted([box.width for box in boxes])
    return np.mean(box_width)


def clean_boxes(boxes):
    mean = get_mean(boxes)
    return [box for box in boxes if box.height > (mean*0.3)]


def erode(boxes, processed):
    boxes.clear()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 6), (1, 1))
    eroded = cv.erode(processed, kernel, iterations=1)
    ctrs, hier = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda contour: cv.boundingRect(contour)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv.boundingRect(ctr)
        boxes.append(Box(x, y, w, h, (0, 255, 0), 2))
    return boxes


def check_erode(boxes, processed):
    mean = get_mean(boxes)
    counter = 0
    for box in boxes:
        if box.width > 80: # > mean*1.5
            counter += 1
    if counter >= 3: # >= 6
        boxes = erode(boxes, processed)
    return boxes


def check_overlap(box1, box2):
    if (box1.x >= (box2.width + box2.x)) or ((box1.width + box1.x) <= box2.x) or ((box1.y + box1.height) <= box2.y) or \
            (box1.y >= (box2.y + box2.height)):
        return False
    else:
        return True


def inside_overlap(box1, box2):
    # Inside if box1 is inside box2
    if (box2.left <= box1.left) and (box2.top <= box1.top) and (box2.right >= box1.right) \
            and (box2.bottom >= box1.bottom):
        return 'inside'
    # Overlap if box1 overlaps box2
    elif check_overlap(box2, box1):
        n = 15
        if (box2.left - n <= box1.left <= box2.left) or (box2.right <= box1.right <= box2.right + n):
            return 'overlap'
    else:
        return False


def get_new_coordinates(box1, box2):
    list_x = sorted([box1.x, box1.bottom_right[0], box2.x, box2.bottom_right[0]])
    list_y = sorted([box1.y, box1.bottom_right[1], box2.y, box2.bottom_right[1]])
    top_left = (list_x[0], list_y[0])
    bottom_right = (list_x[-1], list_y[-1])
    return top_left, bottom_right


def overlap_fix(boxes, image):
    new_boxes = []
    ignore = []
    overlapping = False
    for box in boxes:
        for other_box in boxes:
            if inside_overlap(box, other_box, image) == 'inside' and box != other_box and box not in ignore:
                ignore.append(box)
                overlapping = True
            elif inside_overlap(box, other_box, image) == 'overlap' and box != other_box and box not in ignore:
                top_left, bottom_right = get_new_coordinates(box, other_box)
                new_box = Box(top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1],
                              (255, 0, 255), 1)
                new_boxes.append(new_box)
                ignore.append(box)
                ignore.append(other_box)
                overlapping = True
            else:
                continue
    new_list = new_boxes + [box for box in boxes if box not in ignore]
    sorted_list = sorted(new_list, key=lambda box: box.x)
    return overlapping, sorted_list


def fix_inside_overlapping(boxes, image):
    done = False
    while not done:
        overlapping, boxes = overlap_fix(boxes, image)
        if not overlapping:
            done = True
    return boxes


<<<<<<< HEAD
def divide_box(box, n, j, color):
    new_w = round((box.width / n))
    new_x = round(box.x + (new_w * j))
    new_y = box.y
    new_h = box.height
    new_box = Box(new_x, new_y, new_w, new_h, color, 2)
    return new_box


def check_boxes_to_divide(boxes):
    box_width = sorted([box.width for box in boxes])
    box_height = sorted([box.height for box in boxes])
    mean_width = float(np.mean(box_width))
    mean_height = float(np.mean(box_height))
    mean_rectangle = Box(20, 20, round(mean_width), round(mean_height), (0, 0, 0), 1)
    # boxes.append(mean_rectangle)
    mean = get_mean(boxes)
    box_width = sorted([box.width for box in boxes])

    if box_width[-1] >= mean * 1.85: # *2.2
        new_boxes = []
        boxes_to_remove = []
        for box in boxes:
            n = None
            if round(box.width // (mean*0.9)) >= 2:
                n = round(box.width // mean*0.9)
                color = (255, 0, 0)
                print('Dividing big box')
            elif round(box.width // (mean*0.7)) >= 2:
                n = round(box.width // (mean*0.7))
                color = (255, 125, 125)
                print('Dividing smaller box')
            if n is not None:
                for j in range(int(n)):
                    boxes_to_remove.append(box)
                    new_boxes.append(divide_box(box, n, j, color))

        for box in new_boxes:
            if round(box.width // (mean * 0.7)) >= 2:
                n = round(box.width // (mean * 0.7))
                for j in range(int(n)):
                    boxes_to_remove.append(box)
                    color = (125, 125, 125)
                    print('Dividing again')
                    new_boxes.append(divide_box(box, n, j, color))
        boxes = [box for box in boxes if box not in boxes_to_remove]

        new_list = boxes + new_boxes
        sorted_list = sorted(new_list, key=lambda box: box.x)
        return sorted_list
    return boxes


def show_boxes(boxes, original_boxes, image):
    image_origin = image.copy()
    for box in original_boxes:
        cv.rectangle(image_origin, (box.x, box.y), box.bottom_right, box.color, box.thickness)
    image_origin = cv.resize(image_origin, (image_origin.shape[1] * 3, image_origin.shape[0] * 2))
    cv.imshow('origin', image_origin)

    image_cleaned = image.copy()
    for box in boxes:
        cv.rectangle(image_cleaned, (box.x, box.y), box.bottom_right, box.color, box.thickness)
    image_cleaned = cv.resize(image_cleaned, (image_cleaned.shape[1] * 3, image_cleaned.shape[0] * 2))
    cv.imshow('cleaned', image_cleaned)
    cv.waitKey(0)


def crop_boxes(boxes, image):
    cropped_images = []
    for box in boxes:
        crop_image = image[box.y:(box.y + box.height), box.x:(box.x + box.width)].copy()
        cropped_images.append(crop_image)
    return cropped_images


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
            x_offset = 2
            y_offset = int((32 - height) / 2)

        elif in_width < in_height:
            scale_percent = 28 / in_height
            width = int(cropped_image.shape[1] * scale_percent)
            height = int(cropped_image.shape[0] * scale_percent)
            x_offset = int((32 - width) / 2)
            y_offset = 2

        dim = (width, height)
        cropped_image = cv.resize(cropped_image, dim, interpolation=cv.INTER_AREA)
        blank_image[y_offset:y_offset + cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = \
            cropped_image

        gray = cv.cvtColor(blank_image, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_not(gray)

        edges_list.append(gray)
    return edges_list


def img_segmentation(img_path):
    image = cv.imread(img_path)
    processed = process_img(image)
    original_boxes = get_boxes(processed)
    boxes = fix_inside_overlapping(original_boxes, image)
    boxes = check_boxes_to_divide(boxes)
    boxes = clean_boxes(boxes)
    show_boxes(boxes, original_boxes, cv.imread(img_path))
    cropped_images = crop_boxes(boxes, image)
    pasted_images = image_paste(cropped_images)
    return pasted_images

# for folder in os.listdir('data/lineimages/a01'):
#     for img_path in os.listdir(f'data/lineimages/a01/{folder}'):
#         img_segmentation(f'data/lineimages/a01/{folder}/{img_path}')

img_segmentation('data/lineimages/a01/a01-001/a01-001w-02.tif ')
# img_segmentation('data/lineimages/a01/a01-001/a01-001z-02.tif')
# img_segmentation('data/lineimages/a01/a01-001/a01-001z-03.tif')
# img_segmentation('data/lineimages/a01/a01-000/a01-000x-01.tif')
# img_segmentation('data/lineimages/a01/a01-000/a01-000x-02.tif')
# img_segmentation('data/lineimages/a01/a01-000/a01-000x-03.tif')
# img_segmentation('data/lineimages/a01/a01-000/a01-000x-04.tif')
# img_segmentation('data/lineimages/a01/a01-001/a01-001z-04.tif')

# images = img_segmentation('input4.png')
# for image in images:
#     cv.imshow('image', image)
#     cv.waitKey(0)


# TODO's
# Exprementera med floor eller round division
# Minska standarden man jämför boxarna med
# Slå ihop lådor som ligger ovanför varandra?
# Träna även modellen på stora bokstäver, nu tränas den bara på små.
