##########################################################
# Map segmentation map to curve dict
# Computation Graphic-based Method
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import cv2
import numpy as np
from scipy.signal import convolve2d

def agc_scipy(image, agc_win=5):
    image_mean = convolve2d(image, np.ones((1, agc_win))/agc_win, mode='same', boundary='fill', fillvalue=0)
    image_agc = image / (image_mean+1e-10)
    return image_agc


def interp_curves(curve_dict):
    curve_dict_new = dict()
    for name, curve in curve_dict.items():
        curve = curve[np.argsort(curve[:, 0]), :]
        x_new = np.arange(curve[0, 0], curve[-1, 0])
        y_new = np.interp(x_new, curve[:, 0], curve[:, 1])
        curve_interp = np.array([x_new, y_new]).T.astype(np.int32)
        curve_dict_new[name] = curve_interp
    return curve_dict_new


def rm_overlap(curve_dict, width, overlap_rate=0.7):
    curve_vec, deleted_index = [], []
    for curve in curve_dict.values():
        curve_k_vec = np.zeros(width)
        curve_k_vec[curve[:, 0]-1] = curve[:, 1]-1
        curve_vec.append(curve_k_vec)
    curve_vec_all = np.array(curve_vec)
    curve_dict_cp = curve_dict.copy()
    for k, (name, curve) in enumerate(curve_dict_cp.items()):
        curve_k_mask = np.zeros(width)
        curve_k_mask[curve[:, 0]-1] = curve[:, 1]-1
        curve_vec_all_k_crop = curve_vec_all[:, curve[:, 0]-1]
        curve_k_mask_crop = curve_k_mask[curve[:, 0]-1]
        or_k = np.sum(np.abs(curve_vec_all_k_crop - curve_k_mask_crop) < 5, axis=1)/len(curve_k_mask_crop)
        if np.max(np.delete(or_k, [k] + deleted_index)) > overlap_rate:
            del curve_dict[name]
            deleted_index.append(k)
    return curve_dict


def connect_curves(curve, id, curve_dict, d_const=(3, 8)):
    # * get left points of each curve
    left_points = np.array([curves[0] for curves in curve_dict.values()])
    right_points = np.array([curves[-1] for curves in curve_dict.values()])
    name_list = list(curve_dict.keys())
    
    # * distance between right point of current curve and the left points of curves
    # Vertical distance
    d_v = left_points[:, 1] - curve[-1, 1]
    # horizental distance
    d_h = left_points[:, 0] - curve[-1, 0]
    
    # * condition 1 
    """
    (0 < x^2_l - x^1_r <= d_h)  &  (|y^1_r - y^2_l| <= d_v)
    
           ----------  ----------
    """
    cond1_part1 = (d_h > 0) & (d_h <= d_const[0])
    cond1_part2 = np.abs(d_v) <= d_const[1]
    cond1_ind = np.where(cond1_part1 & cond1_part2)[0]
    if len(cond1_ind)>0:
        sel_id = cond1_ind[np.argmin(np.abs(d_v)[cond1_ind])]
        curve_right = curve_dict[name_list[sel_id]]
        curve_merge = np.vstack((curve, curve_right))
        curve_merge = curve_merge[np.argsort(curve_merge[:, 0]), :]
        return curve_merge, [id, name_list[sel_id]]
    
    # * condition 2 
    """
        (x^1_r - x^2_l <= d_h) & (x^1_r <= x^2_r)  &  (|y^1_r - y^2_l| <= d_v)
        -------------
                -------------
    """
    cond2_part1 = (right_points[:, 0] - curve[-1, 0] >= 0) & (d_h <= 0) & (d_h > -d_const[0])
    cond2_part2 = np.abs(d_v) <= d_const[1]
    cond2_ind = np.where(cond2_part1 & cond2_part2)[0]
    if len(cond2_ind)>0:
        sel_id = cond2_ind[np.argmin(np.abs(d_v)[cond2_ind])]
        curve_right = curve_dict[name_list[sel_id]]
        left_edge_id = np.argmin(np.abs(curve[:, 0]-curve_right[0, 0]))
        right_edge_id = np.argmin(np.abs(curve_right[:, 0]-curve[-1, 0]))
        curve_merge = np.vstack((curve[:left_edge_id], curve_right[right_edge_id:]))
        curve_merge = curve_merge[np.argsort(curve_merge[:, 0]), :]
        return curve_merge, [id, name_list[sel_id]]
    return None


def merge_main(curve_dict, d_const):
    if_merge = 0
    curve_dict_cp = curve_dict.copy()
    for name, curve in curve_dict_cp.items():
        if name in curve_dict.keys():
            del curve_dict[name]
        try:
            merge_info = connect_curves(curve, name, curve_dict, d_const)
        except IndexError:
            merge_info = None
        if merge_info is not None:
            curve_merge, connect_id = merge_info
            del curve_dict[connect_id[1]]
            curve_dict['%s+%s'% tuple(connect_id)] = curve_merge
            if_merge = 1
            return curve_dict, if_merge
        else:
            curve_dict[name] = curve
    return curve_dict, if_merge


# def split_group_ori(seg_map, overlap_rate=0.5, d_const=(3, 5)):
#     # remove the low reasonable pixel
#     seg_map = seg_map / (np.max(seg_map)+1e-5)
#     seg_map[seg_map<0.1] = 0
#     # to uint8 for cv2 processing
#     gray = (seg_map*255).astype('uint8')
    
#     # find the contours
#     contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     curve_list, curve_mean_y = [], []
#     for contour in contours:
#         # making the mask from contours info
#         mask = np.zeros_like(gray)
#         cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)  
#         seg_curve_k = mask.astype(np.float32)
#         curve_y = np.argmax(seg_curve_k, axis=0)
#         curve_x = np.arange(len(curve_y))
#         curve_val = seg_curve_k[curve_y, curve_x]
#         judge_id = curve_val > 0
#         curve_ori = np.array([curve_x[judge_id], curve_y[judge_id]]).T
#         if len(curve_ori) < 2:  # remove single point
#             continue
#         curve_list.append(curve_ori+1)  # +1 means log the real index (not the index in python, i.e., start with 0)
#         curve_mean_y.append(np.mean(curve_ori[:, 1].astype(np.float32)))  # log the mean depth
    
#     # to curve dict
#     order_curve = np.argsort(curve_mean_y)
#     curve_dict = {}
#     for k, id in enumerate(order_curve):
#         curve = curve_list[id]
#         curve_dict[k] = curve[np.argsort(curve[:, 0]), :]
        
#     # remove overlapped curves
#     curve_dict = rm_overlap(curve_dict, seg_map.shape[1], overlap_rate)
    
#     # connect curve lines
#     while 1:
#         curve_dict, if_merge = merge_main(curve_dict, d_const)
#         if if_merge == 0:
#             break
    
#     # save to dict
#     return curve_dict


def split_group(seg_map):
    # remove the low reasonable pixel
    seg_map = seg_map / (np.max(seg_map)+1e-5)
    seg_map[seg_map<0.1] = 0
    # to uint8 for cv2 processing
    gray = (seg_map*255).astype('uint8')
    
    # find the contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    curve_list, curve_mean_y = [], []
    for contour in contours:
        # making the mask from contours info
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED)  
        seg_curve_k = mask.astype(np.float32)
        curve_y = np.argmax(seg_curve_k, axis=0)
        curve_x = np.arange(len(curve_y))
        curve_val = seg_curve_k[curve_y, curve_x]
        judge_id = curve_val > 0
        curve_ori = np.array([curve_x[judge_id], curve_y[judge_id]]).T
        if len(curve_ori) < 2:  # remove single point
            continue
        curve_list.append(curve_ori+1)  # +1 means log the real index (not the index in python, i.e., start with 0)
        curve_mean_y.append(np.mean(curve_ori[:, 1].astype(np.float32)))  # log the mean depth
    
    # to curve dict
    order_curve = np.argsort(curve_mean_y)
    curve_dict = {}
    for k, id in enumerate(order_curve):
        curve = curve_list[id]
        curve_dict[k] = curve[np.argsort(curve[:, 0]), :]
        
    # save to dict
    return curve_dict