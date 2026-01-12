# from torchmetrics import Dice
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, BinaryFBetaScore
from torchmetrics.classification import BinaryJaccardIndex

import numpy as np

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')



#https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
###############################################################################
# metrics used in validation
###############################################################################
class compute_metrics:
    def __init__(self, threshold=0.5, device=0):
        self.tp = threshold
        self.ACC = BinaryAccuracy(threshold=threshold).to(device)
        self.AP = BinaryAveragePrecision(thresholds=5).to(device)#thresholds=5:使用线性间隔的阈值数 0 到 1 作为计算的箱
        #self.MAE = MeanAbsoluteError().to(device)
        # self.dice = Dice(threshold=threshold).to(device)
        self.FBeta = BinaryFBetaScore(beta=2.0, threshold=threshold).to(device)
        self.MIoU = 0.5*BinaryJaccardIndex(threshold=threshold).to(device)  # 添加MIoU指标

    def compute(self, seg_hat, seg_gt):
        seg_hat, seg_gt = seg_hat.squeeze(), seg_gt.squeeze()
        seg_met = self.seg_metrics(seg_hat, seg_gt)
        return seg_met
        
    def seg_metrics(self, seg_hat, seg_gt):
        seg_hat, seg_gt = seg_hat.reshape(-1), seg_gt.reshape(-1).long()
        MetDict = {
            'Seg-ACC': self.ACC(seg_hat, seg_gt).item(),
            'Seg-AP': self.AP(seg_hat, seg_gt).item(), 
            # 'Seg-Dice': self.dice(seg_hat, seg_gt).item(),
            'Seg-FBeta': self.FBeta(seg_hat, seg_gt).item(),
            'Seg-MIoU': self.MIoU(seg_hat, seg_gt).item(),  # 添加MIoU指标
        }
        return MetDict
    
    @staticmethod
    def RCP_metrics(gth, RCP_hat, field_hat, RCP_gt=None, field_gt=None, k=6, thre_rate=0.6):
        # Average Stacking Energy (ASE)
        ase_list = ASE(gth, RCP_hat.values())
        metrics_dict = {
            'ASE': np.mean(ase_list),
            'semblance': semblance(gth, RCP_hat, 3)
        }
        if RCP_gt is not None:
            # Tracking Ratio (TR) & Tracked Average Stacking Energy  (TASE)
            tr, tracked_curve_hat_dict = TR(gth, RCP_gt, RCP_hat, k, thre_rate)
            tase = np.mean(ASE(gth, tracked_curve_hat_dict.values()))
            metrics_dict['TR'] = tr
            metrics_dict['semblance_manual'] = semblance(gth, RCP_gt, 3)
            metrics_dict['TASE'] = tase
            metrics_dict['MSE_field'] = MSE_slope_field(field_hat, field_gt)
            
        return metrics_dict


# Average Stacking Energy (ASE)
def ASE(gth, curve_list):
    se_list = []
    for curve in curve_list:
        se_list.append(np.mean(gth[curve[:, 1]-1, curve[:, 0]-1]))
    return np.array(se_list)


# semblance
def semblance(gth_agc, curve_dict, win_h=3):
    semblance_vals = []
    for curve in curve_dict.values():
        # get the window data
        window_data = np.zeros((2*win_h+1, len(curve)))
        for i, (j, pick) in enumerate(curve):
            try:
                window_data[:, i] = gth_agc[pick-win_h-1:pick+win_h, j-1]
            except ValueError:
                pass
        # 计算每个窗口内的Semblance值
        numerator = np.sum(np.sum(window_data, axis=1)**2)
        denominator = len(curve) * np.sum(np.sum(window_data**2, axis=1))
        
        # 避免除零错误
        semb_k = numerator / denominator if denominator != 0 else 0
        semblance_vals.append(semb_k)
    semblance_mean = np.mean(semblance_vals)
    return semblance_mean

# judge whether track succeed
def judege_track(gth, curve_true, curve_dict_hat, thre_k=2, thre_rate=0.8):
    w = gth.shape[1]
    curve_vec_true = np.zeros(w) 
    curve_vec_true[curve_true[:, 0]-1] = curve_true[:, 1]-1
    for curve_hat in curve_dict_hat.values():
        curve_vec_hat = np.ones(w) * -10000
        curve_vec_hat[curve_hat[:, 0]-1] = curve_hat[:, 1]-1
        true_pick_ind = curve_vec_true !=0
        diff = curve_vec_hat[true_pick_ind]-curve_vec_true[true_pick_ind]
        rate = np.sum(np.abs(diff)<thre_k) / np.sum(true_pick_ind)
        if rate >= thre_rate:
            return True, curve_hat
    return False, []


# Track Ratio (TR) 
def TR(gth, curve_dict_true, curve_dict_hat, thre_k=2, thre_rate=0.8):
    track_condi = []
    track_curve_hat_dict = {}
    for k, curve_true in enumerate(curve_dict_true.values()):
        state, curve_marked = judege_track(gth, curve_true, curve_dict_hat, thre_k, thre_rate)
        track_condi.append(state)
        if state:
            track_curve_hat_dict['tracked_hat_%d'%k] = curve_marked
    return np.mean(track_condi), track_curve_hat_dict


def MSE_slope_field(field_manual, field_auto):
    # MSE
    mse = np.mean((field_manual-field_auto)**2)
    return mse