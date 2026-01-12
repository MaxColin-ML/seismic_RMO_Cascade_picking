import segyio
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

import argparse



def segy2npy(segy_path, save_root, require_list=None):
    # load segy file
    print(segy_path)
    segy_file = segyio.open(segy_path, "r", strict=False)
    os.makedirs(save_root, exist_ok=True)
    # split gather
    cdp_index = np.array(segy_file.attributes(21)[:]).reshape(-1)
    inline_index = np.array(segy_file.attributes(189)[:]).reshape(-1)
    cdp_cut = np.where(np.abs(cdp_index[1:] - cdp_index[:-1]) > 0)[0]
    sample_index = np.array([[0]+list(cdp_cut+1), list(cdp_cut)+[int(segy_file._trace.length)]]).T
    
    # get basic information
    info_dict = {
        't_sample': [int(segy_file.samples[1]-segy_file.samples[0])],
        'trace_num': [int(segy_file._trace.length)],
        'trace_len': [int(segy_file._trace.shape)],
        'gth_num': [int(len(sample_index))]
        }
    info_df = pd.DataFrame(info_dict)
    info_df.to_csv(os.path.join(save_root, 'segy_info.csv'), index=False)
    
    # save gather
    os.makedirs(os.path.join(save_root, 'gth'), exist_ok=True)
    count_bar = tqdm(total=len(sample_index))
    for start, end in sample_index:
        file_name = 'gth_Line%d_Cdp%d' % (inline_index[start], cdp_index[start])
        if require_list is not None:
            if file_name not in require_list:
                count_bar.update(1)
                continue
        traces = segy_file.trace.raw[start: (end+1)]
        traces = traces.astype(np.float32)
        save_name = file_name + '.npy' 
        np.save(os.path.join(save_root, 'gth', save_name), traces)
        del traces
        count_bar.update(1)
    count_bar.close()
    

def bin2npy(bin_folder_path, save_root, h_num=100, w_num=96):
    # load bin file
    file_list = os.listdir(bin_folder_path)
    os.makedirs(save_root, exist_ok=True)

    os.makedirs(os.path.join(save_root, 'gth'), exist_ok=True)
    count_bar = tqdm(total=len(file_list))
    for file_name in file_list:
        line, cdp = file_name.split('_')[1:3]
        line = int(line)
        cdp = int(cdp.split('.')[0])
        with open(os.path.join(bin_folder_path, file_name), 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        matrix = data.reshape((-1, h_num))[:w_num, :]
        svae_file_name = 'gth_Line%d_Cdp%d' % (line, cdp) + '.npy' 
        np.save(os.path.join(save_root, 'gth', svae_file_name), matrix)
        count_bar.update(1)
    count_bar.close()
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segy_path', type=str)
    parser.add_argument('--save_root', type=str)
    parser.add_argument('--parse_sample_txt', type=str, default=None)
    parser.add_argument('--file_type', type=str, default=
                        'segy')
    parser.add_argument('--samp_num', type=int, default=
                        101)
    parser.add_argument('--samp_width', type=int, default=
                        96)
    args = parser.parse_args() 
    
    if args.parse_sample_txt is not None:
        require_list = [line.strip() for line in open(args.parse_sample_txt, 'r')]
    else:
        require_list = None
    if args.file_type == 'segy':
        segy2npy(args.segy_path, args.save_root, require_list)
    else:
        bin2npy(args.segy_path, args.save_root, args.samp_num, args.samp_width)
    
    
    
if __name__ == '__main__':
    main()