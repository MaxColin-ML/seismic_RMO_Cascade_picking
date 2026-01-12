##########################################################
# Quality control method base ASE 
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import numpy as np


def rcp_qc(curve_dict, gth, thre_ase_percent=0.8):
    thre_ase = np.max(gth) * thre_ase_percent
    curve_dict_new = dict()
    for curve_name in curve_dict:
        curve = curve_dict[curve_name]
        se_k = np.mean(gth[curve[:, 1]-1, curve[:, 0]-1])
        if se_k > thre_ase:
            curve_dict_new[curve_name] = curve
    return curve_dict_new