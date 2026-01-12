import itertools
import os
from pathlib import Path


def get_tuning_cmd(base_cmd, tuning_params_dict, fixed_params_dict):
    tuning_names = list(tuning_params_dict.keys())
    comb_list = itertools.product(*tuning_params_dict.values())
    fix_part = ' '.join(['--{} {}'.format(k, v) for k, v in fixed_params_dict.items()])
    cmd_list = []
    for comb in comb_list:
        tuning_comb = ' '.join(['--{} {}'.format(tuning_names[k], v) for k, v in enumerate(list(comb))])
        cmd_list.append('%s %s %s ' % (base_cmd, fix_part, tuning_comb))
    return cmd_list


def assign_tuning_cmd(base_cmd, assign_keys, assign_comb_list, fixed_params_dict, tuning_params_dict):
    tuning_names = list(tuning_params_dict.keys())
    tuning_list = itertools.product(*tuning_params_dict.values())
    comb_list = itertools.product(tuning_list, assign_comb_list)
    cmd_list = []
    for tuning_cmd, assign_cmd in comb_list:
        if len(tuning_cmd) == 0:
            tuning_comb = ''
        else:
            tuning_comb = ' '.join(['--{} {}'.format(tuning_names[k], v) for k, v in enumerate(list(tuning_cmd))])
        assign_comb = ' '.join(['--{} {}'.format(assign_keys[k], v) for k, v in enumerate(list(assign_cmd))])
        cmd_list.append(tuning_comb+' '+assign_comb)
    fix_part = ' '.join(['--{} {}'.format(k, v) for k, v in fixed_params_dict.items()])
    final_cmd_list = []
    for cmd in cmd_list:
        final_cmd_list.append('%s %s %s ' % (base_cmd, fix_part, cmd))
    return final_cmd_list


def divide_group(list_cmd, group_num):
    if group_num == 1:
        return [list_cmd]
    else:
        elm_num = len(list_cmd) // group_num
        groups_list = []
        for i in range(0, len(list_cmd), elm_num):
            try:
                groups_list.append(list_cmd[i:i + elm_num])
            except IndexError:
                groups_list.append(list_cmd[i:])
        return groups_list

def mk_sh_file(sh_root, excute_cmd_list):
    os.makedirs(os.path.dirname(sh_root), exist_ok=True)
    path = Path(sh_root)
    with path.open('w', newline='', encoding='utf-8') as f:
        for cmd in excute_cmd_list:
            f.write('{}\n'.format(cmd))
        
        
if __name__ == "__main__":
    para_dict = {
        'lr': [1e-2, 1e-3, 1e-4],
        'bs': [2, 3, 4],
        'loss': ['L2', 'L1', 'W']
    }
    fix_dict = {
        'device': 0,
        'save_group': 1
    }
    cmd_list = get_tuning_cmd('python train.py', para_dict, fix_dict, 'test', 'log')
    print(cmd_list)
    for cmds in divide_group(cmd_list, 4):
        print(len(cmds))