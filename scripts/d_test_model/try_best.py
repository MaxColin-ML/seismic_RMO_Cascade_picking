import os
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from utils.tuning_tool import assign_tuning_cmd, divide_group, mk_sh_file

# python scripts/ablation/run_ablation.py
"""
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_model/sh_file/test_model_241231-1.sh >scripts/test_model/log/test-1.log 2>&1 & # 106998
ps aux | grep 'your_script.sh'
"""


base_name = 'test_model_241231'
base_cmd = 'python models/cascade_method/train_run.py'
sh_root = 'scripts/test_model'
split_num = 1

para_keys = ['first_act', 'dcn_use', 'cbam_use', 'add_peak', 'add_agc', 'add_bp']

comb_list = [
    # first_act dcn_use cbam_use add_peak add_agc add_bp
    ['tanh', 1, 1, 1, 1, 1],
    # ['relu', 1, 1, 1, 1, 1],
    # ['tanh', 0, 1, 1, 1, 1],
    # ['tanh', 1, 0, 1, 1, 1],
    # ['tanh', 1, 1, 0, 1, 1],
    # ['tanh', 1, 1, 1, 0, 1],
    # ['tanh', 1, 1, 1, 1, 0],
]

fix_dict = {
    # model parameters
    'seg_net': 'MSFSegNet',
    'agc_list': '31+51',
    'CBAM_red': 16,
    # training parameters
    'lr_init': 1e-3,
    'train_bs': 16,
    'optimizer': 'Adam',
    'loss_func': 'BCE',
    'seed': 1,
    # path setting
    'save_group': base_name,
    'device': 0,
    'if_print': 0,
}

os.makedirs(os.path.join(sh_root, 'log'), exist_ok=True)
cmd_list = assign_tuning_cmd(base_cmd, para_keys, comb_list, fix_dict)
for id, cmds in enumerate(divide_group(cmd_list, split_num)):
    mk_sh_file(os.path.join(sh_root, 'sh_file/%s-%d.sh'%(base_name, id+1)), cmds)

