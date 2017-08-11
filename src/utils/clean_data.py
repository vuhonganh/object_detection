import sys
import bbox_reader
import os
# get the argument
try:
    data_path = sys.argv[1]
except IndexError:
    print('use default')
    data_path = '/data/hav16/imagenet/'
print('data path is %s' % data_path)

cur_dir = os.path.dirname(os.path.realpath(__file__))
dict_wnid_name = bbox_reader.get_class_name_dict(cur_dir + '/class_name.txt')
annot_path = data_path + '/Annotation/'
dest_file = data_path + '/all_bbox.txt'
dest_clean_file = data_path + '/clean_bbox.txt'
bbox_reader.generate_img_bbox(annot_path, dest_file, dict_wnid_name)
bbox_reader.clean_data(data_path, dest_file, dest_clean_file)

print('\ncleaning data done\n')