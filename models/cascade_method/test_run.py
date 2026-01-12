##########################################################
# Test code
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import argparse

from config.data_path import save_root, data_dict
from torch.utils.data import DataLoader

from models.cascade_method.train_util import training_loop
from utils.data_loader import dataloader_lab
from models.cascade_method.seg_nets import MSFSegNet, UNet


def test_main():
    # ============= parameter setting =================
    parser = argparse.ArgumentParser()
    # path setting
    # ablation_loss_opt_250125/
    parser.add_argument('--save_root', type=str, default=save_root)
    parser.add_argument('--model_group', type=str, default='ablation_loss_opt_250125')
    parser.add_argument('--model_name', type=str, default='MSFSegNet-31+51-32-tanh-01111-bs8-lr1.0e-04-BCE-SGD-S2')
    #ablation_250102/MSFSegNet-31+51-16-relu-01111-bs16-lr1.0e-03-BCE-Adam-S1
    # test setting
    parser.add_argument('--test_set', type=str, default='sh')
    # feature setting
    parser.add_argument('--new_agc_list', type=str, default='15+31')
    # posterior processing setting
    parser.add_argument('--win_k', type=int, default=4)
    parser.add_argument('--bw_data', type=float, default=5.0)
    parser.add_argument('--bw_para', type=float, default=50.0)
    parser.add_argument('--valid_range', type=int, default=50)
    # clustering setting
    parser.add_argument('--clu_eps', type=float, default=4.0)
    # output setting
    parser.add_argument('--min_len', type=int, default=6)
    # computation setting
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_num', type=int, default=8)
    parser.add_argument('--proc_num', type=int, default=8)
    args = parser.parse_args() 
    
    # =========== preparing test ===============
    # base path & name
    test_set_path = data_dict[args.test_set]
    save_name = 'test-%s-AGC=%s-rg=%d+%.1f+%.1f+%d-cluK=%.1f-%d' % (args.test_set, args.new_agc_list, args.win_k, args.bw_data, args.bw_para, args.valid_range, args.clu_eps, args.min_len)
    config = args
    new_agc_list = list(map(int, args.new_agc_list.split('+')))
    # define segmentation network
    # args.save_name = '%s-%s-%d-%s-%d%d%d%d%d-bs%d-lr%.1e-%s-%s-S%d' % (
        # args.seg_net 0, 
        # args.agc_list 1, args.CBAM_red 2, args.first_act 3, 
        # args.dcn_use 4-1, args.cbam_use 4-2,
        # args.add_peak 4-3, args.add_agc 4-4, args.add_bp 4-5, 
        # args.train_bs, args.lr_init, 
        # args.loss_func, args.optimizer, args.seed
        # 
    model_info = config.model_name.split('-')[:5]
    if model_info[0] == 'MSFSegNet':
        #输入的通道数为 1
        picker = MSFSegNet(agc_list=new_agc_list, CBAM_reduction=int(model_info[2]), basic_act=model_info[3], dcn_use=int(model_info[4][0]), cbam_use=int(model_info[4][1]), add_peak=int(model_info[4][2]), add_bp=int(model_info[4][4]), device=config.device)
    elif model_info[0] == 'UNet':
        picker = UNet(agc_list=new_agc_list, add_peak=int(model_info[4][2]), device=config.device)
    else:
        raise ValueError("Error: invalid segmentation network")
    picker.cuda(config.device)

    # define framework
    train_class = training_loop(
        root_path=os.path.join(config.save_root, config.model_group, config.model_name), 
        optimizer=None, 
        opt_strategy=None,
        threshold=0.1,
        model=picker,  
        train_dataloader=None, 
        valid_dataloader=None, 
        criterion=None,
        max_epoch=20, 
        early_stop_patience=4, 
        config=config, 
        device=config.device,
        seed=1, 
        training_name=save_name
    )
    
    # =========== starting test ===============
    # load test dataloader
    sample_list = [file_n.strip('lab_').strip('.npy') for file_n in os.listdir(os.path.join(test_set_path, 'lab'))]
    test_dl = DataLoader(
        dataloader_lab(test_set_path, sample_list=sample_list),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)
    test_hyper_dict = {
        'win_k': config.win_k, 
        'bw_data': config.bw_data, 
        'bw_para': config.bw_para, 
        'valid_range': config.valid_range, 
        'min_len': config.min_len, 
        'clu_eps': config.clu_eps, 
        'proc_num': config.proc_num
        }
    # start test
    train_class.test_step(test_dl, new_agc_list, save_name, config.save_num, test_hyper_dict)
    
if __name__ == "__main__":
    test_main()