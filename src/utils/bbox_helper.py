import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread, imresize, imsave, imshow


def bbox_parser(bbox_info_file, data_dir_path='/data/hav16/imagenet/'):
    """
    parse the bbox info text file into a dictionary
    :param bbox_info_file: full-path file containing bbox info, e.g. /data/hav16/imagenet/clean_bbox.txt
    :return: dictionary
    """
    all_info = {}  # a dict by image name, each info is a dict itself
    nb_img_per_class = {}  # a dict by class name, map each class to number of times it appears in dataset
    class_to_idx = {}  # map a class name to a digit label
    with open(bbox_info_file, mode='r') as f:
        for line in f:
            line = line.replace('\n', '')
            info_list = line.split(',')


def string_to_bbox(string_info):
    """string_info has syntax: file_name, xmin1, ymin1, xmax1, ymax1,  object1_name\n 
    file_name, img_width, img_height, xmin2, ymin2, xmax2, ymax2, object2_name\n... etc."""
    list_line = string_info.split('\n')
    bbs = []
    start_idx = 3
    for line in list_line:
        if line == '':
            continue
        list_info = line.split(',')

        xmin = int(list_info[start_idx])
        ymin = int(list_info[start_idx + 1])
        xmax = int(list_info[start_idx + 2])
        ymax = int(list_info[start_idx + 3])
        obj_name = list_info[start_idx + 4]
        bbs.append([xmin, ymin, xmax, ymax, obj_name])
    return bbs


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