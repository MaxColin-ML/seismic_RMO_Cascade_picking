import numpy as np
import os
import torch.utils.data as data
from scipy.signal import find_peaks
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

# trace-wise normalization 
def trace_wise_norm(data):
    return data / (np.max(np.abs(data), axis=0)+1e-10)


class dataloader_lab(data.Dataset):
    """
    dataset iterator for the sliced samples
    """
    def __init__(self, root_path, min_win=16, sample_list=None) -> None:
        super().__init__()
        self.min_win = min_win
        self.samp_root = os.path.join(root_path, 'gth')
        self.label_root = os.path.join(root_path, 'lab')
        self.cache_root = os.path.join(root_path, 'cache')
        os.makedirs(self.cache_root, exist_ok=True)
        if sample_list is None:
            self.index_list = [name.strip('.npy').strip('gth_') for name in os.listdir(self.samp_root)]
        else:
            self.index_list = sample_list

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, ind):
        samp_id = self.index_list[ind]
        m = self.min_win
        # load gather
        gth = self.get_RCP_gth(samp_id)
        try:
            lab = self.get_manual_lab(samp_id)
        except:
            lab = {}
        h, w = gth.shape
        h_new = (h//m+1)*m if h%m > 0 else h
        w_new = (w//m+1)*m if w%m > 0 else w
        gth_pad = np.zeros((h_new, w_new), dtype=np.float32)
        gth_pad[:h, :w] = gth
        gth_peak_map = np.zeros((h_new, w_new), dtype=np.float32)
        if os.path.exists(os.path.join(self.cache_root, 'gth_%s_peak.npy'%samp_id)):
            gth_peak_map[:h, :w] = np.load(os.path.join(self.cache_root, 'gth_%s_peak.npy'%samp_id))
        else:
            gth_peak_map[:h, :w] = self.get_peak_map(gth)
            np.save(os.path.join(self.cache_root, 'gth_%s_peak.npy'%samp_id), gth_peak_map[:h, :w])
        gth_feat = np.concatenate([gth_pad[np.newaxis, ...], gth_peak_map[np.newaxis, ...]], axis=0)
        # generate ground truth mask
        if lab:
            mask_gt = self.make_mask(gth_pad, lab, [h_new, w_new])
        else:
            mask_gt = np.zeros_like(gth_pad)
            
        return gth_feat, mask_gt, samp_id, '%d-%d' % (h, w)

    def get_RCP_gth(self, samp_id):
        gth = np.load(os.path.join(self.samp_root, 'gth_%s.npy'%samp_id)).T.astype(np.float32)
        return gth
    
    def get_manual_lab(self, samp_id):
        lab = np.load(os.path.join(self.label_root, 'lab_%s.npy'%samp_id), allow_pickle=True).item()
        for key, value in lab.items():
            lab[key] = np.array(value)
        return lab
    
    @staticmethod
    def make_mask(gth, curve_dict, shape_const):
        mask_gt = np.zeros_like(gth)
        for _, curve in curve_dict.items():
            curve_array = np.array(curve).astype(np.int32)
            curve_remain = curve_array[(curve_array[:, 1]<=shape_const[0]) & (curve_array[:, 0]<=shape_const[1]), :]
            mask_gt[curve_remain[:, 1]-1, curve_remain[:, 0]-1] = 1
        return mask_gt
    
    @staticmethod
    def process_lab_txt(path_txt):
        with open(path_txt, 'r') as file:
            curve_data = eval(file.read())  
        for key, value in curve_data.items():
            curve_data[key] = np.array([list(map(int, point)) for point in value])
        return curve_data
    
    @staticmethod
    def get_peak_map(gth):
        peak_map = np.zeros_like(gth)
        for j, trace in enumerate(gth.T):
            peaks, _ = find_peaks(trace, height=0)
            peak_map[peaks, j] = 1
        return peak_map



import numpy as np
# load the picking results of semblance-based method
def process_lab_txt(path_txt):
    with open(path_txt, 'r') as file:
        curve_data = eval(file.read())  
    for key, value in curve_data.items():
        curve = np.array([list(map(int, point)) for point in value])
        curve_data[key] = curve
    return curve_data
