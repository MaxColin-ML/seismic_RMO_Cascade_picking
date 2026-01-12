##########################################################
# Training code
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import os
import sys

import pandas as pd

import torch
import torch.nn as nn

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import argparse

from config.data_path import data_root, save_root, data_dict
from torch.utils.data import DataLoader

from models.cascade_method.train_util import training_loop, set_seed
from utils.data_loader import dataloader_lab
from models.cascade_method.seg_nets import MSFSegNet, UNet
from models.cascade_method.loss_func import mix_loss

def train_main():
    parser = argparse.ArgumentParser()
    #########################################
    # path setting
    #########################################
    parser.add_argument('--parse_root', type=str, default=data_root)
    parser.add_argument('--save_root', type=str, default=save_root)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--save_group', type=str, default='tuning_train_para_250105')
 
    #########################################
    # model setting
    #########################################
    parser.add_argument('--seg_net', type=str, default='MSFSegNet')
    # tuning hyper-parameters
    # model
    parser.add_argument('--agc_list', type=str, default='31+51')
    parser.add_argument('--CBAM_red', type=int, default=16)
    parser.add_argument('--first_act', type=str, default='tanh')
    # training
    parser.add_argument('--lr_init', type=float, default=1e-3)
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--loss_func', type=str, default='BCE')
    parser.add_argument('--seed', type=int, default=1)
    # ablation part
    parser.add_argument('--dcn_use', type=int, default=0)
    parser.add_argument('--cbam_use', type=int, default=1)
    parser.add_argument('--add_peak', type=int, default=1)
    parser.add_argument('--add_agc', type=int, default=1)
    parser.add_argument('--add_bp', type=int, default=1)
    
    
    
    #########################################
    # test setting
    #########################################
    parser.add_argument('--if_test', type=int, default=1)
    parser.add_argument('--save_num', type=int, default=4)
    
    #########################################
    # other setting
    #########################################
    # device setting
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--if_print', type=int, default=1)
    args = parser.parse_args() 
    
    ######################################################################
    # 载入训练、验证、测试数据集路径
    ######################################################################
    train_set = data_dict['syn_us_train']
    valid_set = data_dict['syn_us_valid']
    args.save_name = '%s-%s-%d-%s-%d%d%d%d%d-bs%d-lr%.1e-%s-%s-S%d' % (
        args.seg_net, 
        args.agc_list, args.CBAM_red, args.first_act, 
        args.dcn_use, args.cbam_use,
        args.add_peak, args.add_agc, args.add_bp, 
        args.train_bs, args.lr_init, 
        args.loss_func, args.optimizer, args.seed
        ) if args.save_name is None else args.save_name
    valid_bs = args.train_bs
    
    # load the training parameters
    config = args
    agc_list = list(map(int, args.agc_list.split('+')))
    # define segmentation network
    set_seed(config.seed)
    if config.seg_net == 'MSFSegNet':
        picker = MSFSegNet(agc_list=agc_list, CBAM_reduction=config.CBAM_red, basic_act=config.first_act, dcn_use=config.dcn_use, cbam_use=config.cbam_use, add_peak=config.add_peak, add_bp=config.add_bp, device=config.device)
    elif config.seg_net == 'UNet':
        picker = UNet(agc_list=agc_list, add_peak=config.add_peak, device=config.device)
    else:
        raise ValueError("Error: invalid segmentation network")
    picker.cuda(config.device)

    ######################################################################
    # train processs
    ######################################################################
    train_dl0 = dataloader_lab(train_set)
    valid_dl0 = dataloader_lab(valid_set)
    #torch的dataloader
    set_seed(config.seed)
    train_dl = DataLoader(
        train_dl0,
        batch_size=config.train_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    valid_dl = DataLoader(
        valid_dl0,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True)
    
    # loss
    if config.loss_func == 'BCE':
        criterion = nn.BCELoss()
    elif config.loss_func == 'mix':
        criterion = mix_loss(config.device)
    else:
        raise ValueError("Error: invalid loss function")
    
    # optimizer
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(picker.parameters(), lr=config.lr_init)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(picker.parameters(), lr=config.lr_init, momentum=0.9)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(picker.parameters(), lr=config.lr_init, weight_decay=1e-6)

    else:
        raise ValueError("Error: invalid optimizer") 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config.lr_init/10)
    
    # define train process
    folder_root = os.path.join(config.save_root, config.save_group, config.save_name)
    train_class = training_loop(
        root_path=folder_root, 
        optimizer=optimizer, 
        opt_strategy=scheduler,
        threshold=config.threshold,
        model=picker,  
        train_dataloader=train_dl, 
        valid_dataloader=valid_dl, 
        criterion=criterion,
        max_epoch=40, 
        early_stop_patience=5, 
        config=config, 
        device=config.device,
        seed=config.seed, 
        training_name=config.save_name, 
        if_print=config.if_print
    )
    train_class.training_process()

    ############################################################
    # test process
    ############################################################
    if config.if_test:
        test_results = []
        test_dict = {
            # 'syn_us_test': ['31+51', 3, 5, 50, 50, 2, 8],
            'bp': ['9+15', 3, 5, 50, 50, 2, 8],
            'yz': ['15+31', 3, 5, 50, 50, 2, 8],
            'sh': ['15+31', 3, 5, 50, 50, 2, 8],
            'jsh': ['15+31', 3, 5, 50, 50, 2, 8],
        }
        for name, (agc_new, win_k, bw_data, bw_para, valid_range, clu_eps, min_len) in test_dict.items():
            save_name = 'test-%s-AGC=%s-rg=%d+%.1f+%.1f+%d-cluK=%.1f-%d' % (name, agc_new, win_k, bw_data, bw_para, valid_range, clu_eps, min_len)
            hyper_para = {
                'win_k': win_k, 
                'bw_data': bw_data, 
                'bw_para': bw_para, 
                'valid_range': valid_range, 
                'min_len': min_len, 
                'clu_eps': clu_eps,
                'proc_num': 4
            }
            print('# > Task: %s' % save_name)
            sample_list = [file_n.strip('lab_').strip('.npy') for file_n in os.listdir(os.path.join(data_dict[name], 'lab'))]
            test_dl = DataLoader(
                dataloader_lab(data_dict[name], sample_list=sample_list),
                batch_size=valid_bs,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True)
            agc_list = list(map(int, agc_new.split('+')))
            # save to list
            test_df = train_class.test_step(test_dl, agc_list, save_name, config.save_num, hyper_para)
            values = test_df.values
            columns = list(test_df.columns)
            columns_df = ['name'] + columns
            test_results.append([name]+list(values[0]))
            
        # summary results to xlsx
        test_sum_df = pd.DataFrame(test_results, columns=columns_df)
        test_sum_df.to_csv(os.path.join(folder_root, 'test_summary.csv'), index=False)
            
if __name__ == "__main__":
    train_main()