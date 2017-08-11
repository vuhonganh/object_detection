import sys
import bbox_reader

# get the argument
try:
    data_path = sys.argv[1]
except IndexError:
    print('use default')
    data_path = '/data/hav16/imagenet/'

print('data path is %s' % data_path)
list_obj_names = bbox_reader.get_list_obj_names('class_name.txt')
annot_path = data_path + '/Annotation/'
dest_file = data_path + '/all_bbox.txt'
dest_clean_file = data_path + '/clean_img_bbox.txt'
bbox_reader.generate_img_bbox(annot_path, dest_file, list_obj_names)
bbox_reader.clean_data(data_path, dest_file, dest_clean_file)

print('\ncleaning data done\n')