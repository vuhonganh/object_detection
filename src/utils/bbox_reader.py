"""
This file helps to preprocess images from imagenet with bounding box
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread, imresize, imsave, imshow
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


def process_xml_annotation(xml_file, list_object_name):
    """Find all bbox that contains the object by matching object_name. 
    Note that an annotation folder can contain xml files of different folder name
    and multiple objects    
    Return empty string if bbox is invalid and/or does not contain any object in list_object_name
    if list_object_name is None, return a string containing all valid boxes
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
        if list_object_name is None or cur_obj in list_object_name:
            bb = ob.find('bndbox')
            xmin = int(float(bb.find('xmin').text))
            ymin = int(float(bb.find('ymin').text))
            xmax = int(float(bb.find('xmax').text))
            ymax = int(float(bb.find('ymax').text))
            if box_is_valid(xmin, ymin, xmax, ymax, img_width, img_height):
                bbs.append([xmin, ymin, xmax, ymax])
                string_res += '%s,%d,%d,%d,%d,%s\n' % (img_file_name, xmin, ymin, xmax, ymax, cur_obj)
                if list_object_name is not None and folder_name not in list_object_name:
                    set_other_idx.add(folder_name)
    return string_res


def string_to_bbox(string_info):
    """string_info has syntax: file_name, xmin1, ymin1, xmax1, ymax1,  object1_name\n 
    file_name, xmin2, ymin2, xmax2, ymax2, object2_name\n... etc."""
    list_line = string_info.split('\n')
    bbs = []
    for line in list_line:
        if line == '':
            continue
        list_info = line.split(',')

        xmin = int(list_info[1])
        ymin = int(list_info[2])
        xmax = int(list_info[3])
        ymax = int(list_info[4])
        obj_name = list_info[5]
        bbs.append([xmin, ymin, xmax, ymax, obj_name])
    return bbs


def process_folder_xml(folder_path, dest_file, list_object_names):
    """extract bbox info from all xml files in the folder_path and write to dest_file.
    If list_object_names is provided (i.e. not None) then only bboxes containing one of those objects are extracted.
    bbox info has format: img_file_name,xmin,ymin,xmax,ymax,object_name"""
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    with open(dest_file, mode='w') as f:
        for xml_f in xml_files:
            string_info = process_xml_annotation(folder_path + '/' + xml_f, list_object_names)
            if string_info != '':
                f.write(string_info)


def test_write_file():
    folder_path = '../../samples'
    list_object_names = None
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
    with open('test_img_bbox.txt', mode='w') as f:
        for xml_f in xml_files:
            string_info = process_xml_annotation(folder_path + '/' + xml_f, list_object_names)
            if string_info != '':
                f.write(string_info)


def visualize_bbox(file_img, file_xml, list_object_name=None):
    bbs = string_to_bbox(process_xml_annotation(file_xml, list_object_name))
    img = imread(file_img)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(bbs)):
        xmin = bbs[i][0]
        ymin = bbs[i][1]
        xmax = bbs[i][2]
        ymax = bbs[i][3]
        obj_name = bbs[i][4]
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin + 3, ymin, obj_name, bbox=dict(facecolor='yellow', alpha=0.6))
    plt.show()


def bbox_vis_example():
    # file_img = '../../samples/n02870526_10384.JPEG'
    # file_xml = '../../samples/n02870526_10384.xml'
    # file_img = '../../samples/n07739125_12.JPEG'
    # file_xml = '../../samples/n07739125_12.xml'
    file_img = '../../samples/n02773037_9927.JPEG'
    file_xml = '../../samples/n02773037_9927.xml'

    # list_object_name = ('n02870526', 'n07739125')
    visualize_bbox(file_img, file_xml)


def generate_img_bbox(annotation_path='/data/hav16/imagenet/Annotation/', dest_file='all_bbox.txt', list_object_names=None):
    with open(dest_file, mode='w') as res_file:
        for d in os.listdir(annotation_path):
            folder_path = annotation_path + '/' + d
            xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
            for xml_f in xml_files:
                string_info = process_xml_annotation(folder_path + '/' + xml_f, list_object_names)
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


def _clean_data():
    path_to_all = '/data/hav16/imagenet/'
    print('removing images without bbox')
    remove_no_bbox_imgs(path_to_all)
    print('write clean bbox info file')
    write_clean_img_bbox(path_to_all)


def get_list_obj_names():
    id_to_name = {}
    with open('class_name.txt') as f:
        for line in f:
            pair = line.split(' ')
            id_to_name[pair[0]] = pair[1]
    list_obj_names = list(id_to_name.keys())
    return list_obj_names

if __name__ == '__main__':
    # bbox_vis_example()
    # generate_img_bbox(list_object_names=get_list_obj_names())
    # remove_no_bbox_imgs(list_img, path_to_imgs)
    # find_lacking_imgs(list_img, path_to_imgs)
    # write_clean_img_bbox()
    # test_write_file()
    _clean_data()