import os
import sys
import itertools

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from config.data_path import save_root
from utils.tuning_tool import assign_tuning_cmd, divide_group, mk_sh_file


# predict samples with our casdcade method
focus_set = ['bp']
para_keys = ['new_agc_list', 'win_k', 'bw_data', 'bw_para', 'min_len', 'clu_eps']
samp_list_dict = {set_name: os.path.join('/home/htwang/research/seismic/RMO_picking/Cascade_RMO_Code/scripts/ep_tuning_pred_hyperp/log', 'samp_list_%s.npy'%set_name) for  set_name in focus_set}
mark_name = 'pred=hpyerp_test'
#                agc  win_k bw_data bw_para min_len clu_eps 
#                 0      1     2       3        4       5
default_comb = ['9+15', 5,    5,       50,     15,     12]
comb_list = [default_comb]

# agc feature 
# for k1, k2 in itertools.product([3, 5, 7, 9], [15, 21, 25, 31, 51]):
#     comb_k = default_comb.copy()
#     comb_k[0] = '%d+%d'%(k1, k2)
#     comb_list.append(comb_k)
    
# win_k 
# for win_k in [18, 20, 25, 30]:   # [3, 7, 9, 12, 15]
#     comb_k = default_comb.copy()
#     comb_k[1] = win_k
#     comb_list.append(comb_k) 
    
# # bw_data
# for bw_data in [3, 7, 9, 12, 15]: 
#     comb_k = default_comb.copy()
#     comb_k[2] = bw_data
#     comb_list.append(comb_k) 
    
# # bw_para
# for bw_para in [30, 40, 50, 60, 70, 80, 90, 100]: 
#     comb_k = default_comb.copy()
#     comb_k[3] = bw_para
#     comb_list.append(comb_k) 

# min_len
for min_len in [20, 25, 30, 35, 40, 45, 50]: 
    comb_k = default_comb.copy()
    comb_k[4] = min_len
    comb_list.append(comb_k) 
    
    
# # clu_eps
# for clu_eps in [8, 10, 12, 15, 18, 20]: 
#     comb_k = default_comb.copy()
#     comb_k[5] = clu_eps
#     comb_list.append(comb_k) 
    
    
base_name = 'hpyerp_test_250305'
base_cmd = 'python models/cascade_method/predict_run.py'
sh_root = 'scripts/ep_tuning_pred_hyperp'
split_num = 3

fix_dict = {
    'model_group': 'z_best_250207',
    'model_name': 'MSFSegNet-31+51-32-tanh-01011-bs8-lr1.0e-04-BCE-SGD-S1',
    'proc_num': 4,
    'device': 0,
    'samp_list_npy': '/home/htwang/research/seismic/RMO_picking/Cascade_RMO_Code/scripts/ep_tuning_pred_hyperp/log/samp_list_bp.npy',
    'mark_name': mark_name,
    'test_set': 'bp',
    'valid_range': 50
}

os.makedirs('scripts/ep_tuning_pred_hyperp/sh_file', exist_ok=True)
cmd_list = assign_tuning_cmd(base_cmd, para_keys, comb_list, fix_dict, {})
for id, cmds in enumerate(divide_group(cmd_list, split_num)):
    mk_sh_file('scripts/ep_tuning_pred_hyperp/sh_file/%s_%d.sh'%(base_name, id+1), cmds)
    
"""
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_pred_hyperp/sh_file/hpyerp_test_250305_1.sh > scripts/ep_tuning_pred_hyperp/log/hpyerp_test_250305_1.log 2>&1 & # 2096651
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_pred_hyperp/sh_file/hpyerp_test_250305_2.sh > scripts/ep_tuning_pred_hyperp/log/hpyerp_test_250305_2.log 2>&1 & # 2096786
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_pred_hyperp/sh_file/hpyerp_test_250305_3.sh > scripts/ep_tuning_pred_hyperp/log/hpyerp_test_250305_3.log 2>&1 & # 2096651
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ep_tuning_pred_hyperp/sh_file/hpyerp_test_250305_4.sh > scripts/ep_tuning_pred_hyperp/log/hpyerp_test_250305_4.log 2>&1 & # 2096786
"""
