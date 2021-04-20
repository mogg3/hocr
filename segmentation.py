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
        boxes.append(Rectangle(x, y, w, h))

    boxes = check_erode(boxes)
    return boxes


def clean_boxes(boxes):
    return [box for box in boxes if box.height > 5 and box.width > 5]


def check_erode(boxes):
    counter = 0
    for box in boxes:
        if box.width > 80:
            counter += 1
    if counter >= 3:
        print("Eroding")
        boxes.clear()
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 6), (1, 1))
        eroded = cv.erode(processed, kernel, iterations=1)
        ctrs, hier = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv.boundingRect(ctr)
            boxes.append(Rectangle(x, y, w, h))
            # cv.rectangle(image,(x,y),( x + w, y + h ),(0,0,255),2)
    return boxes


def check_overlap(R1, R2):
    if (R1.x >= (R2.width + R2.x)) or ((R1.width + R1.x) <= R2.x) or ((R1.y + R1.height) <= R2.y) or (R1.y >= (R2.y + R2.height)):
        return False
    else:
        return True


def get_overlapping_boxes(box, boxes):
    overlaps = []
    for i in range(len(boxes)):
        if check_overlap(box, boxes[i]) and box != boxes[i]:
            overlaps.append(boxes[i])
    return overlaps


def inside(R1, R2):
    if (R1.x < R2.x) and (R1.y < R2.y) and (R1.x + R1.width > R2.x + R2.width) and (R1.y + R1.height > R2.y + R2.height):
        return True
    else:
        return False


def get_dicts(boxes):
    return {box: get_overlapping_boxes(box, boxes) for box in boxes if get_overlapping_boxes(box, boxes)}


def show_boxes(boxes):
    for box in boxes:
        cv.rectangle(image, (box.x, box.y), box.bottom_right, (0, 0, 255), 1)
    cv.imshow('marked areas', image)
    cv.waitKey(0)


def remove_inside_boxes():
    for box in boxes:
        for box_ in boxes:
            if inside(box, box_) and box != box_:
                boxes.remove(box_)
    return boxes


def divide_boxes(mean):
    big_boxes = [box for box in boxes if box.width >= mean * 1.3]
    for box in big_boxes:
        print(f'{box.width} // {mean * 0.7} = {box.width // (mean * 0.7)}')
        # Divide boxes
        cv.rectangle(image, (box.x, box.y), box.bottom_right, (255, 0, 0), 1)

    cv.imshow('marked areas', image)
    cv.waitKey(0)


def check_box_width():
    box_width = sorted([box.width for box in boxes])
    mean = np.mean(box_width)
    if box_width[-1] >= mean*2.2:
        divide_boxes(mean)


img_path = 'input3.tif'
image = cv.imread(img_path)

processed = process_img(image)
boxes = get_boxes(processed)
boxes = clean_boxes(boxes)
boxes = remove_inside_boxes()
check_box_width()
show_boxes(boxes)
