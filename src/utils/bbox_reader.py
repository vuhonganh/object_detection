"""
This file helps to preprocess images from imagenet with bounding box
"""

import xml.etree.ElementTree as ET
import os


set_other_idx = set()

def _get_int(name, root, box_idx=0):
    """ find item with that name in the box with that box_idx then return its value in int"""
    cur_idx = 0
    for item in root.iter(name):
        if cur_idx == box_idx:
            return int(float(item.text))
        cur_idx += 1  # go to next bounding box
    # found no box with that box_idx
    return -1


def _get_number_bbox(root):
    nb_box = 0
    while True:
        if _get_int('xmin', root, nb_box) == -1:
            break
        nb_box += 1
    return nb_box


def box_is_valid(xmin, ymin, xmax, ymax, width, height):
    if xmin <= xmax <= width and ymin <= ymax <= height:
        return True
    return False


def process_xml_annotation(xml_file, class_name_dict):
    """Find all bbox that contains the object by matching object_name. 
    Note that an annotation folder can contain xml files of different folder name
    and multiple objects    
    Return empty string if bbox is invalid and/or does not contain any object in class_name_dict
    if class_name_dict is None, return a string containing all valid boxes
    class_name_dict[wnid] -> english name of that wnid
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    folder_name = root.findtext('folder')
    img_file_name = root.findtext('filename') + '.JPEG'
    img_width = float(root.find('size').find('width').text)
    img_height = float(root.find('size').find('height').text)

    # find all bounding boxes that contain object in list_object_name
    bbs = []
    string_res = ''

    for ob in root.findall('object'):
        cur_obj = ob.find('name').text
        if len(class_name_dict) == 0 or cur_obj in class_name_dict:
            bb = ob.find('bndbox')
            xmin = int(float(bb.find('xmin').text))
            ymin = int(float(bb.find('ymin').text))
            xmax = int(float(bb.find('xmax').text))
            ymax = int(float(bb.find('ymax').text))
            if box_is_valid(xmin, ymin, xmax, ymax, img_width, img_height):
                bbs.append([xmin, ymin, xmax, ymax])
                if cur_obj in class_name_dict:
                    cur_obj = class_name_dict[cur_obj]  # change wnid to english name
                string_res += '%s,%d,%d,%d,%d,%d,%d,%s\n' % (img_file_name, img_width, img_height, xmin, ymin, xmax, ymax, cur_obj)
                if len(class_name_dict) > 0 and folder_name not in class_name_dict:
                    set_other_idx.add(folder_name)
    return string_res


def process_folder_xml(folder_path, dest_file, class_name_dict):
    """extract bbox info from all xml files in the folder_path and write to dest_file.
    If list_object_names is provided (i.e. not None) then only bboxes containing one of those objects are extracted.
    bbox info has format: img_file_name,im_width,im_height,xmin,ymin,xmax,ymax,object_name"""
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    with open(dest_file, mode='w') as f:
        for xml_f in xml_files:
            string_info = process_xml_annotation(folder_path + '/' + xml_f, class_name_dict)
            if string_info != '':
                f.write(string_info)


def test_write_file():
    folder_path = '../../samples'
    class_name_dict = {}
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    with open('test_img_bbox.txt', mode='w') as f:
        for xml_f in xml_files:
            string_info = process_xml_annotation(folder_path + '/' + xml_f, class_name_dict)
            if string_info != '':
                f.write(string_info)


def generate_img_bbox(annotation_path='/data/hav16/imagenet/Annotation/', dest_file='all_bbox.txt', class_name_dict={}):
    with open(dest_file, mode='w') as res_file:
        for d in os.listdir(annotation_path):
            folder_path = annotation_path + '/' + d
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            for xml_f in xml_files:
                string_info = process_xml_annotation(folder_path + '/' + xml_f, class_name_dict)
                # print(string_info)
                if string_info != '':
                    res_file.write(string_info)

    print(set_other_idx)


def get_imgs_having_bbox(bbox_info_file):
    list_img = set()
    with open(bbox_info_file) as f:
        for line in f:
            if line != '':
                img_name = line.split(',')[0]
                list_img.add(img_name)
    return list(list_img)


def remove_no_bbox_imgs(path_to_all_imgs, bbox_info_file='all_bbox.txt'):
    """note that there are a lot of images without bbox -> remove them"""
    # first find all images in path_to_imgs:
    all_img_files = [e for e in os.listdir(path_to_all_imgs) if e.endswith('.JPEG')]
    imgs_with_bbox = get_imgs_having_bbox(bbox_info_file)
    for e in all_img_files:
        if e not in imgs_with_bbox:
            os.remove(path_to_all_imgs + '/' + e)


def write_clean_img_bbox(path_to_all_imgs, bbox_info_file='all_bbox.txt', clean_bbox_info_file='clean_bbox.txt'):
    """note that there are lacking images, some annotated images are not found in .tar file"""
    all_img_files = [e for e in os.listdir(path_to_all_imgs) if e.endswith('.JPEG')]
    # first find all lacking images
    lacking_imgs = []
    imgs_having_bbox = get_imgs_having_bbox(bbox_info_file)
    for e in imgs_having_bbox:
        if e not in all_img_files:
            lacking_imgs.append(e)
    # write clean bbox info file
    with open(clean_bbox_info_file, mode='w') as clean_file:
        with open(bbox_info_file) as all_file:
            for line in all_file:
                if line != '':
                    img_name = line.split(',')[0]
                    if img_name not in lacking_imgs:
                        clean_file.write(line)


def clean_data(path_to_all_imgs, bbox_info_file, clean_bbox_info_file):
    print('removing images without bbox')
    remove_no_bbox_imgs(path_to_all_imgs, bbox_info_file)
    print('write clean bbox info file')
    write_clean_img_bbox(path_to_all_imgs, bbox_info_file, clean_bbox_info_file)


def get_class_name_dict(class_name_file='class_name.txt'):
    class_name_dict = {}
    with open(class_name_file) as f:
        for line in f:
            line = line.replace('\r', '').replace('\n', '')
            pair = line.split(',')
            class_name_dict[pair[0]] = pair[1]
    return class_name_dict
