"""
This is helper file to implement region proposal network (rpn)
some code is based on Yhenon's implementation https://github.com/yhenon/keras-frcnn
original paper is FasterRCNN: https://arxiv.org/abs/1506.01497
Note that for Region Proposal Network:
Inputs: CNN Feature Map.
Outputs: A bounding box per anchor. A score representing how likely the image in that bounding box will be an object.
We then pass each such bounding box that is likely to be an object into Fast R-CNN's head to generate a 
classification and tightened bounding boxes.
"""

from bbox_helper import get_bbox_list_resized, bbox_parser, random_visualize_bbox_img, show_img_with_bbox, show_img_from_file
import numpy as np
import random
np.set_printoptions(threshold=np.inf)  # set this to force numpy to fully print

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


def iou_dict(bb1_dict, bb2_dict):
    # note that list form is the following: xmin, ymin, xmax, ymax
    bb1 = [bb1_dict['xmin'], bb1_dict['ymin'], bb1_dict['xmax'], bb1_dict['ymax']]
    bb2 = [bb2_dict['xmin'], bb2_dict['ymin'], bb2_dict['xmax'], bb2_dict['ymax']]
    return iou(bb1, bb2)

def get_resized_img_size(width, height, resized_img_min_size):
    if width < height:
        new_height = int(height * float(resized_img_min_size) / width)
        new_width = resized_img_min_size
    else:
        new_width = int(width * float(resized_img_min_size) / height)
        new_height = resized_img_min_size
    return new_width, new_height


def compute_feat_size_resnet(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length
    return get_output_length(width), get_output_length(height)


def compute_regr(anchor_box, gt_box):
    # implement second part of equation 2 in Faster-RCNN paper
    anchor_width = anchor_box[2] - anchor_box[0]
    anchor_height = anchor_box[3] - anchor_box[1]
    gt_center_x = 0.5 * (gt_box[0] + gt_box[2])
    gt_center_y = 0.5 * (gt_box[1] + gt_box[3])
    anchor_center_x = 0.5 * (anchor_box[0] + anchor_box[2])
    anchor_center_y = 0.5 * (anchor_box[1] + anchor_box[3])

    tx_star = (gt_center_x - anchor_center_x) / anchor_width
    ty_star = (gt_center_y - anchor_center_y) / anchor_height
    tw_star = np.log((gt_box[2] - gt_box[0]) / anchor_width)
    th_star = np.log((gt_box[3] - gt_box[1]) / anchor_height)
    return tx_star, ty_star, tw_star, th_star


def compute_rpn_of_img(img_info, config, width, height, resized_width, resized_height, compute_feature_sizes):
    """
    
    :param img_info: all_info from bbox_parser but for each image only
    :param config: a dict contain some attribute for configuration
    :param width: original width of the image
    :param height: original height of the image
    :param resized_width: resized width of the image
    :param resized_height: resized height of the image
    :param compute_feature_sizes: width and height of the output when passing resized image through the conv layers
    :return: rpn of that image
    """
    # step 0.1: get info from config file
    down_scale = config['down_scale']  # scale downed after conv layers (4 time 2-stride = 2^4 = 16)
    anchor_sizes = config['anchor_sizes']
    anchor_ratios = config['anchor_ratios']
    n_anchors = len(anchor_sizes) * len(anchor_ratios)

    # step 0.2: prepare data from img_info
    n_bbs = len(img_info['bbox'])  # number of bboxes in this image
    best_iou_for_bb = np.zeros(n_bbs, dtype=np.float32)
    n_anchor_ratios = len(config['anchor_ratios'])

    # step 0.3 init some buffers
    best_anchor_for_bb = -1 * np.ones((n_bbs, 4), dtype=np.int32)
    best_corner_for_bb = np.ones((n_bbs, 4), dtype=np.int32)
    best_param_for_bb = np.ones((n_bbs, 4), dtype=np.float32)
    num_anchor_for_bb = np.zeros(n_bbs, dtype=np.int32)

    # step 0.4: resize bbox:
    ground_truth_bb_list = get_bbox_list_resized(img_info['bbox'], width, height, resized_width, resized_height)

    # step 1: compute a number of anchors for each pixel in the output mapget_bbox_list_resized
    # step 1.1: get feature map size (after conv layers), note that image input is the resized one
    feat_width, feat_height = compute_feature_sizes(resized_width, resized_height)

    # step 1.2: init output objectives at every pixel of feature map, consider n anchors
    y_is_box_valid = np.zeros((feat_height, feat_width, n_anchors))  # take binary value, 1 means anchor is in image
    y_rpn_overlap = np.zeros((feat_height, feat_width, n_anchors))  # take binary value, 1 means anchor is positive
    y_rpn_regr = np.zeros((feat_height, feat_width, n_anchors * 4))  # describe 4 params for regression (tx, ty, th, tw)

    # step 1.3: for each pixel in feature map, compute valid anchors
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(len(anchor_ratios)):
            anchor_width = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_height = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
            for feat_x in range(feat_width):
                # from center of the anchor box, up_scale and +- anchor_width/2 to get xmin, xmax
                # + 0.5 since idx starts from 0
                anchor_xmin = down_scale * (feat_x + 0.5) - (anchor_width / 2)
                anchor_xmax = down_scale * (feat_x + 0.5) + (anchor_width / 2)
                if anchor_xmin < 0 or anchor_xmax > resized_width:
                    continue
                for feat_y in range(feat_height):
                    anchor_ymin = down_scale * (feat_y + 0.5) - (anchor_height / 2)
                    anchor_ymax = down_scale * (feat_y + 0.5) + (anchor_height / 2)
                    if anchor_ymin < 0 or anchor_ymax > resized_height:
                        continue
                    # now this is a valid anchor, by default set its type to negative
                    anchor_type = 'negative'

                    best_iou_for_loc = 0.0  # for current location on feat map + current anchor configuration
                    # different from best IOU for a ground truth box
                    # init best_regression for current location
                    best_regression = (0.0, 0.0, 0.0, 0.0)  # not sure if initialize by 0. is a good idea

                    for bb_idx in range(n_bbs):
                        # compute iou of the current ground truth box vs the current anchor box
                        # current ground truth bbox
                        cur_gt = [ground_truth_bb_list[bb_idx]['xmin'], ground_truth_bb_list[bb_idx]['ymin'],
                                  ground_truth_bb_list[bb_idx]['xmax'], ground_truth_bb_list[bb_idx]['ymax']]
                        cur_anchor = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
                        cur_iou = iou(cur_gt, cur_anchor)
                        if cur_iou > best_iou_for_bb[bb_idx] or cur_iou > config['upper_bound_iou']:
                            # implement second part of equation 2 in Faster-RCNN paper
                            tx_star, ty_star, tw_star, th_star = compute_regr(cur_anchor, cur_gt)

                            # anchor is considered positive only when it's > upper bound
                            if cur_iou > config['upper_bound_iou']:
                                anchor_type = 'positive'
                                num_anchor_for_bb[bb_idx] += 1
                                if cur_iou > best_iou_for_loc:
                                    best_iou_for_loc = cur_iou
                                    best_regression = (tx_star, ty_star, tw_star, th_star)

                            if config['lower_bound_iou'] < cur_iou < config['upper_bound_iou']:
                                anchor_type = 'neutral'  # will not participate to objective

                            if cur_iou > best_iou_for_bb[bb_idx]:
                                best_anchor_for_bb[bb_idx] = [feat_y, feat_x, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bb[bb_idx] = cur_iou
                                best_corner_for_bb[bb_idx] = [anchor_xmin, anchor_xmax, anchor_ymin, anchor_ymax]
                                best_param_for_bb[bb_idx] = [tx_star, ty_star, tw_star, th_star]

                    if anchor_type == 'negative':
                        y_is_box_valid[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0
                    elif anchor_type == 'positive':
                        y_is_box_valid[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        y_rpn_overlap[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
                        # because of 4 params for regression
                        start_idx = 4 * (anchor_ratio_idx + n_anchor_ratios * anchor_size_idx)
                        print(best_regression)
                        y_rpn_regr[feat_y, feat_x, start_idx: start_idx + 4] = best_regression
                    else:
                        y_is_box_valid[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0
                        y_rpn_overlap[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 0


    # every ground truth bbox must have at least one positive anchor:
    for bb_idx in range(n_bbs):
        if num_anchor_for_bb[bb_idx] == 0:
            print('bbox with no positive anchor')
            if best_anchor_for_bb[bb_idx, 0] == -1:
                continue
            feat_y, feat_x, anchor_ratio_idx, anchor_size_idx = best_anchor_for_bb[bb_idx]
            y_is_box_valid[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
            y_rpn_overlap[feat_y, feat_x, anchor_ratio_idx + n_anchor_ratios * anchor_size_idx] = 1
            start_idx = 4 * (anchor_ratio_idx + n_anchor_ratios * anchor_size_idx)
            y_rpn_regr[feat_y, feat_x, start_idx: start_idx + 4] = best_param_for_bb[bb_idx]

    # arrange output by (anchor_idx, feat_y, feat_x) (i.e.: anchor_idx, row_idx, col_idx)
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))

    # expand_dims make the shape become (1, anchor_idx, row_idx, col_idx)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])
    print('num pos = ', num_pos)
    # print(pos_locs[0])
    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions. (see part 3.1.3 in paper)
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        to_ignore_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][to_ignore_locs], pos_locs[1][to_ignore_locs], pos_locs[2][to_ignore_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        to_ignore_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) + num_pos - num_regions)
        y_is_box_valid[0, neg_locs[0][to_ignore_locs], neg_locs[1][to_ignore_locs], neg_locs[2][to_ignore_locs]] = 0

    y_rpn_class = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)
    print(np.repeat(y_rpn_overlap, 4, axis=1))
    # y_rpn_class: 1, 2 * nb_anchors, feat_height, feat_width
    # y_rpn_regr: 1, 8 * nb_anchors, feat_height, feat_width
    # TODO: check why stack like this?

    # print(y_is_box_valid)
    # print(y_is_box_valid.shape)

    return y_rpn_class, y_rpn_regr


def get_all_anchor(resized_img_width, resized_img_height, config):
    # first get feature map sizes:
    feat_width, feat_height = compute_feat_size_resnet(resized_img_width, resized_img_height)
    down_scale = config['down_scale']
    anchor_sizes = config['anchor_sizes']
    anchor_ratios = config['anchor_ratios']
    # n_anchors = len(anchor_sizes) * len(anchor_ratios)
    anchors = []
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(len(anchor_ratios)):
            anchor_width = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_height = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
            for feat_x in range(feat_width):
                # from center of the anchor box, up_scale and +- anchor_width/2 to get xmin, xmax
                # + 0.5 since idx starts from 0
                anchor_xmin = down_scale * (feat_x + 0.5) - (anchor_width / 2)
                anchor_xmax = down_scale * (feat_x + 0.5) + (anchor_width / 2)
                if anchor_xmin < 0 or anchor_xmax > resized_img_width:
                    continue
                for feat_y in range(feat_height):
                    anchor_ymin = down_scale * (feat_y + 0.5) - (anchor_height / 2)
                    anchor_ymax = down_scale * (feat_y + 0.5) + (anchor_height / 2)
                    if anchor_ymin < 0 or anchor_ymax > resized_img_height:
                        continue
                    cur_anchor = {'class': 'anchor', 'xmin': anchor_xmin, 'ymin': anchor_ymin,
                                  'xmax': anchor_xmax, 'ymax': anchor_ymax}
                    anchors.append(cur_anchor)
    return anchors


if __name__ == '__main__':
    print(compute_feat_size_resnet(600, 600))
    im_w, im_h = 500, 500
    config = {'down_scale': 16, 'anchor_sizes': [64, 128, 256], 'anchor_ratios': [[1, 1], [1, 2], [2, 1], [2, 2]],
              'upper_bound_iou': 0.65, 'lower_bound_iou': 0.3}
    resized_width, resized_height = get_resized_img_size(im_w, im_h, 600)

    orig_ground_truth_bb = [{'class': 'banana', 'xmin': 123, 'ymin': 23, 'xmax': 333, 'ymax': 465}]

    gt_bbs = get_bbox_list_resized(orig_ground_truth_bb, 500, 500, resized_width, resized_height)
    all_anchors = get_all_anchor(resized_width, resized_height, config)

    print(len(all_anchors))
    all_ious = [iou_dict(gt_bbs[0], all_anchors[i]) for i in range(len(all_anchors))]
    idx_sorted = np.argsort(all_ious)

    print(idx_sorted[-3:])

    all_bbs = [all_anchors[i] for i in idx_sorted[-3:]]
    print(all_bbs)
    all_bbs += gt_bbs
    print(all_bbs)

    all_infos, _, _ = bbox_parser('/data/hav16/imagenet/clean_bbox.txt')
    img_info = all_infos[0]
    print(img_info)
    # show_img_from_file('/data/hav16/imagenet/n07753592_11949.JPEG', all_bbs, resized_width, resized_height)
    # img_width = img_info['width']
    # img_height = img_info['height']

    # random_visualize_bbox_img(all_infos, 0)
    y_class, y_regr = compute_rpn_of_img(img_info, config, im_w, im_h, resized_width, resized_height, compute_feat_size_resnet)
    print(y_class.shape)
    print(y_regr.shape)
    # print(y_regr)