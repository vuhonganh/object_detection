import xml.etree.ElementTree as ET


def get_int(name, root, box_idx=0):
    cur_idx = 0
    for item in root.iter(name):
        if cur_idx == box_idx:
            return int(float(item.text))
        cur_idx += 1  # go to next bounding box
    # found no box with that box_idx
    return -1


def get_number_bbox(root):
    nb_box = 0
    while True:
        if get_int('xmin', root, nb_box) == -1:
            break
        nb_box += 1
    return nb_box


def box_is_valid(xmin, ymin, xmax, ymax, width, height):
    if xmin <= xmax <= width and ymin <= ymax <= height:
        return True
    return False


def process_xml_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    folder_name = root.findtext('folder')
    img_file_name = root.findtext('filename') + '.JPEG'
    img_width = float(root.find('size').find('width').text)
    img_height = float(root.find('size').find('height').text)

    # find all bounding boxes that have same name as folder_name
    bbs = []
    for ob in root.findall('object'):
        if folder_name == ob.find('name').text:
            bb = ob.find('bndbox')
            xmin = int(float(bb.find('xmin').text))
            ymin = int(float(bb.find('ymin').text))
            xmax = int(float(bb.find('xmax').text))
            ymax = int(float(bb.find('ymax').text))
            if box_is_valid(xmin, ymin, xmax, ymax, img_width, img_height):
                bbs.append([xmin, ymin, xmax, ymax])

    print("file name is %s" % img_file_name)
    for bb in bbs:
        print(bb)
    return bbs

bbs = process_xml_annotation('../../n02870526_14674.xml')
