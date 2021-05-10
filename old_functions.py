def get_overlapping_boxes(box, boxes):
    overlaps = []
    for i in range(len(boxes)):
        if check_overlap(box, boxes[i]) and box != boxes[i]:
            overlaps.append(boxes[i])
    return overlaps


def get_dicts(boxes):
    return {box: get_overlapping_boxes(box, boxes) for box in boxes if get_overlapping_boxes(box, boxes)}