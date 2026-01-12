import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils.tuning_tool import get_tuning_cmd, divide_group, mk_sh_file

base_name = 'tuning_train_para_250105'
base_cmd = 'python models/cascade_method/train_run.py'
sh_root = 'scripts/tuning_parameters'
split_num = 6


para_dict = {
    'lr_init': [1e-2, 1e-3, 2e-3, 5e-3, 1e-4],
    'train_bs': [8, 16, 32],
    'CBAM_red': [16, 32],
    'seed': [1, 2, 3, 4, 5],
}

fix_dict = {
    'save_group': base_name,
    'seg_net': 'MSFSegNet',
    'device': 0,
    'first_act': 'tanh',
    'dcn_use': 0,
    'cbam_use': 1,
    'add_peak': 1,
    'add_agc': 1,
    'add_bp': 1,
    'if_print': 0,
    'optimizer': 'SGD',
    'loss_func': 'BCE'
}

for folder in ['sh_file', 'log']:
    os.makedirs(os.path.join(sh_root, folder), exist_ok=True)
cmd_list = get_tuning_cmd(base_cmd, para_dict, fix_dict)
root_path = '/home/htwang/research/seismic/trend_pick/cluster_seg_net'
for id, cmds in enumerate(divide_group(cmd_list, split_num)):
    mk_sh_file('%s/sh_file/%s_%d.sh'%(sh_root, base_name, id+1), cmds)

# python scripts/tuning_parameters/tuning.py
"""
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_parameters/sh_file/tuning_train_para_250105_3.sh > scripts/ep_tuning_parameters/log/tutuning_train_para_250105_3.log 2>&1 & # 2449166
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_parameters/sh_file/tuning_train_para_250105_4.sh > scripts/ep_tuning_parameters/log/tutuning_train_para_250105_4.log 2>&1 & # 1255989
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_parameters/sh_file/tuning_train_para_250105_5.sh > scripts/ep_tuning_parameters/log/tutuning_train_para_250105_5.log 2>&1 & # 3977343
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/ep_tuning_parameters/sh_file/tuning_train_para_250105_6.sh > scripts/ep_tuning_parameters/log/tutuning_train_para_250105_6.log 2>&1 & # 484528
"""
