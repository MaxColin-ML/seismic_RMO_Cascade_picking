##########################################################
# Merge the similar curvatures
# DBSCAN base on a novel distance defination
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

from sklearn.cluster import DBSCAN
import numpy as np


def dist_func(c1, c2, k1, k2, algha=0.5):
    para_w = np.array([1, 4])
    c1_l, c1_r = c1[[0, -1], :]
    c2_l, c2_r = c2[[0, -1], :]
    d_min = min(np.linalg.norm((c1_l-c2_r)*para_w), np.linalg.norm((c1_r-c2_l)*para_w))
    if np.isnan(k1) or np.isnan(k2):
        dist = d_min 
    else:
        dist = (1-algha) * d_min + algha*np.abs(k1-k2)
    return dist


def clustering_main(curve_dict, slope_dict, eps=4):
    curves_info = [[curves, slope_dict[name]] for name, curves in curve_dict.items()]
    n_curve = len(curves_info)
    mat_dist = np.zeros((n_curve, n_curve))
    for i in range(n_curve):
        for j in range(i + 1, n_curve):
            dist = dist_func(curves_info[i][0], curves_info[j][0], curves_info[i][1], curves_info[j][1])
            mat_dist[i, j] = dist
            mat_dist[j, i] = dist
    labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit_predict(mat_dist)
    curve_dict_concat = dict()
    for id in np.unique(labels):
        curve_dict_concat[id] = np.concatenate([curves_info[i][0] for i in range(n_curve) if labels[i] == id])
    return curve_dict_concat, labels