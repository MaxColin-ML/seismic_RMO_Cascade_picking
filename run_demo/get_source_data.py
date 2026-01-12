import os
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from models.cascade_method.get_curve import interp_curves


def process_lab_txt(path_txt):
    with open(path_txt, 'r') as file:
        curve_data = eval(file.read())  
    for key, value in curve_data.items():
        curve_data[key] = np.array([list(map(int, point)) for point in value])
    return curve_data


def get_single_sample(root_path, sample_name):
    # load basic info
    basic_df = pd.read_csv(os.path.join(root_path, 'segy_info.csv'))
    # load CIG 
    CIG_data = np.load(os.path.join(root_path, 'gth', 'gth_%s.npy'%sample_name))
    # load auto picking
    curve_m = np.load(os.path.join(root_path, 'lab', 'lab_%s.npy'%sample_name), allow_pickle=True).item()
    curve_m = interp_curves(curve_m)
    # load semblance picking
    curve_s = process_lab_txt(os.path.join(root_path, 'pic_txt', 'lab_%s.npy'%sample_name))
    curve_s = interp_curves(curve_s)
    samp_info_dict = {
        'CIG_data': CIG_data,
        'curve_m': curve_m,
        'curve_s': curve_s,
        'basic_df': basic_df
    }
    return samp_info_dict