import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.misc import imread, imresize, imsave, imshow
import numpy as np

def bbox_parser(bbox_info_file, data_dir_path='/data/hav16/imagenet/'):
    """
    parse the bbox info text file into a dictionary
    :param bbox_info_file: full-path file containing bbox info, e.g. /data/hav16/imagenet/clean_bbox.txt
    it's csv format: is file_name (without full path), width, height, xmin, ymin, xmax, ymax, class_name
    :param data_dir_path: path to the data directory
    :return: a list of dictionary's value, a dict of nb img per class, a dict of class to idx
    """
    all_info = {}  # a dict by image name, each info is a dict itself
    nb_img_per_class = {}  # a dict by class name, map each class to number of times it appears in dataset
    class_to_idx = {}  # map a class name to a digit label

    if data_dir_path[-1] != '/':
        data_dir_path += '/'

    with open(bbox_info_file, mode='r') as f:
        for line in f:
            if line == '':
                continue
            line = line.replace('\n', '')  # strip new line character
            info_list = line.split(',')

            # update nb image per class and mapping class to idx
            class_name = info_list[7]
            if class_name not in nb_img_per_class:
                nb_img_per_class[class_name] = 1
            else:
                nb_img_per_class[class_name] += 1
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            file_name = info_list[0]

            if file_name not in all_info:  # init the dict for each file_name
                all_info[file_name] = {}
                all_info[file_name]['width'] = int(info_list[1])
                all_info[file_name]['height'] = int(info_list[2])
                all_info[file_name]['file_path'] = data_dir_path + info_list[0]
                all_info[file_name]['bbox'] = []  # a list of bbox, each bbox is a dict itself

            all_info[file_name]['bbox'].append({'class': class_name, 'xmin': int(info_list[3]), 'ymin': int(info_list[4]),
                                                  'xmax': int(info_list[5]), 'ymax': int(info_list[6])})
    return list(all_info.values()), nb_img_per_class, class_to_idx


def random_visualize_bbox_img(list_all_info, idx_to_show=None, show_hflip=False, new_width_r=1.0, new_height_r=1.0):
    if idx_to_show is None:
        idx_to_show = np.random.randint(len(list_all_info))
    im = imread(list_all_info[idx_to_show]['file_path'])
    bb = list_all_info[idx_to_show]['bbox']
    show_img_with_bbox(im, bb)
    imr, bbr, imf, bbf = None, None, None, None
    if new_height_r != 1.0 or new_width_r != 1.0:
        imr, bbr = get_scaled_img(im, bb, new_width_r, new_height_r)
        show_img_with_bbox(imr, bbr)
    if show_hflip:
        imf, bbf = get_hflip_img(im, bb)
        show_img_with_bbox(imf, bbf)
    plt.show()

def show_img_with_bbox(img, bbox_list):
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(bbox_list)):
        xmin = int(bbox_list[i]['xmin'])
        xmax = int(bbox_list[i]['xmax'])
        ymin = int(bbox_list[i]['ymin'])
        ymax = int(bbox_list[i]['ymax'])
        obj_name = bbox_list[i]['class']
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin+1, ymin, obj_name, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.show(block=False)


def get_hflip_img(img, bbox_list):
    height, width, n_channel = img.shape  # suppose img is loaded in this format, note n_row is height
    img_flip = np.fliplr(img)
    new_bbox_list = []
    for i in range(len(bbox_list)):
        ymin = int(bbox_list[i]['ymin'])
        ymax = int(bbox_list[i]['ymax'])
        xmax = width - int(bbox_list[i]['xmin'])
        xmin = width - int(bbox_list[i]['xmax'])
        new_bbox_list.append({'class': bbox_list[i]['class'], 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    return img_flip, new_bbox_list


def get_scaled_img(img, bbox_list, new_width_r, new_height_r):
    height, width, n_channel = img.shape  # suppose img is loaded in this format, note n_row is height
    new_height = int(height * new_height_r)
    new_width = int(width * new_width_r)

    img_resized = imresize(img, (new_height, new_width))
    new_bbox_list = []
    for i in range(len(bbox_list)):
        new_xmin = int(int(bbox_list[i]['xmin']) * new_width_r)
        new_xmax = int(int(bbox_list[i]['xmax']) * new_width_r)
        new_ymin = int(int(bbox_list[i]['ymin']) * new_height_r)
        new_ymax = int(int(bbox_list[i]['ymax']) * new_height_r)
        cur = {'class': bbox_list[i]['class'], 'xmin': new_xmin, 'xmax': new_xmax, 'ymin': new_ymin, 'ymax': new_ymax}
        new_bbox_list.append(cur)
    return img_resized, new_bbox_list


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


if __name__ == '__main__':
    list_all_info, _, _ = bbox_parser('/data/hav16/imagenet/clean_bbox.txt')
    random_visualize_bbox_img(list_all_info, show_hflip=True, new_height_r=0.6, new_width_r=0.8)