""" 
2024/7/18
改进加反向合成数据的方法

author: yang yang (xjtu)
revised: hongtao wang (xjtu)
"""
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import os
import json
import argparse

import numpy as np

from tqdm import tqdm
from models.generate_syn_data.syn_data_func import synthetic_data, add_gaussian_noise


def main():
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('--path_root', type=str, default='/data/wht/seismic/RCP/synthetic_data_ours_train_p')
    # synthetic hyperparameters
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--samp_h', type=int, default=1000)
    parser.add_argument('--samp_w', type=int, default=100)
    parser.add_argument('--num_curve', type=int, default=40)
    parser.add_argument('--angle_noise', type=int, default=1)
    parser.add_argument('--start_id', type=int, default=1)
    args = parser.parse_args() 
    
    gth_path_root = os.path.join(args.path_root, 'gth')
    lab_path_root = os.path.join(args.path_root, 'lab')
    os.makedirs(gth_path_root, exist_ok=True)
    os.makedirs(lab_path_root, exist_ok=True)
    
    # control parameters
    h = args.samp_h
    w = args.samp_w
    k = args.num_curve # curve number
    beta_range_neg = (-0.045, -0.015)
    beta_range_pos = (0.011, 0.045)
    gamma_range_neg = (-0.000025, -0.000015)
    gamma_range_pos = (0.000011, 0.000025)
    freq_range = (60, 200) # main frequency range
    scale = 0.3 # std of gaussian noise 
    
    log_bar = tqdm(total=args.num_samples, colour='#8ecae6')
    for i in range(args.start_id, args.start_id+args.num_samples+1):
        gather, label = synthetic_data(k, h, w, beta_range_neg, beta_range_pos, gamma_range_neg, gamma_range_pos, freq_range, add_angle_noise=args.angle_noise)
        
        # add gaussian noise
        noisy_data = add_gaussian_noise(gather, scale=scale)
        noisy_data[gather==0] = 0
        
        np.save(os.path.join(gth_path_root, 'gth_Line0_Cdp%d.npy'%i), noisy_data.T)
        np.save(os.path.join(lab_path_root, 'lab_Line0_Cdp%d.npy'%i), label)
        log_bar.update(1)
        log_bar.set_description_str('Generating %d/%d'%(i-args.start_id, args.num_samples))


if __name__ == "__main__":
    main()
    """
    # test cmd
    python models\generate_syn_data\syn_data_main.py --path_root E:\各种项目\我的东方物探项目\剩余曲率拾取\数据集\synthetic_data_test2 --num_samples 50
    """