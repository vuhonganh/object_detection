


# Note that bbox represented by xmin, ymin, xmax, ymax
# use list instead of dict for faster comptutation -> bb_list has form xmin, ymin, xmax, ymax

def area(bb):
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


def intersection(bb1, bb2):
    # top left corner of the intersection is computed by
    x_top_left = max(bb1[0], bb2[0])
    y_top_left = max(bb1[1], bb2[1])
    # bottom right subtracted by top left to get width, height size
    width_intersection = min(bb1[2], bb2[2]) - x_top_left
    height_intersection = min(bb1[3], bb2[3]) - y_top_left
    if width_intersection < 0 or height_intersection < 0:
        return 0
    return width_intersection * height_intersection


def union(bb1, bb2):
    # union = area1 + area2 - intersection
    return area(bb1) + area(bb2) - intersection(bb1, bb2)


def iou(bb1, bb2):
    # intersection over union
    return float(intersection(bb1, bb2)) / float(union(bb1, bb2) + 1e-7)  # avoid dividing by zero


def get_new_img_size(width, height, new_img_min_size):
    if width < height:
        new_height = int(height * float(new_img_min_size) / width)
        new_width = new_img_min_size
    else:
        new_height = new_img_min_size
        new_width = int(width * float(new_img_min_size) / height)

    return new_width, new_height


def compute_rpn():
    pass