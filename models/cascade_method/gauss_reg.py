##########################################################
# Estimating the robust curvatures
# Bayesian Kernel Regression Method
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################

import numpy as np
from tqdm import tqdm


def linear_regression(X, y):
    """
    线性回归的闭式解
    X: 特征矩阵 (n_samples, n_features)
    y: 目标值 (n_samples,)
    返回: 参数向量 beta
    """
    # 在 X 中添加一列全为 1 的偏置项
    X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # 计算闭式解
    beta = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    return beta


def gaussian_kernel(x, xi, bandwidth):
    """
    计算高斯核权重
    x: 当前点
    xi: 样本点
    bandwidth: 核宽度
    """
    return np.exp(-np.sum((x - xi) ** 2) / (2 * bandwidth ** 2))

def constrained_kernel_regression(x, X, y, bandwidth, slope_mean, slope_variance, alpha):
    """
    带正态分布约束的核回归（只约束斜率参数）
    x: 预测点 (d,)
    X: 样本点矩阵 (n, d)
    y: 样本点目标值 (n,)
    bandwidth: 核宽度
    slope_mean: 斜率部分的均值 (d,)
    slope_variance: 斜率部分的方差
    alpha: 正态分布约束强度
    返回: 预测值
    """
    n, d = X.shape

    # 构造加权矩阵
    weights = np.array([gaussian_kernel(x, X[i], bandwidth) for i in range(n)])
    W = np.diag(weights)

    # 扩展样本点矩阵以适应线性回归模型
    X_augmented = np.hstack((np.ones((n, 1)), X))  # 添加偏置项

    # 构造正态分布约束矩阵（偏置项不受约束）
    constraint = np.zeros((d + 1, d + 1))
    constraint[1:, 1:] = alpha / (2 * slope_variance) * np.eye(d)  # 只对斜率部分施加约束
    shift_term = np.zeros(d + 1)
    shift_term[1:] = alpha / slope_variance * slope_mean

    # 正则化核回归系数计算
    A = X_augmented.T @ W @ X_augmented  + constraint
    b = X_augmented.T @ W @ y + shift_term
    theta = np.linalg.solve(A, b)
    
    # 对预测点进行扩展
    x_augmented = np.hstack(([1], x))
    
    # 返回预测值
    return x_augmented @ theta


class posterior_regression:
    def __init__(self, win_k, bw_data, bw_para):
        self.win_k = win_k
        self.bw_data = bw_data
        self.bw_para = bw_para

    def est_prior(self, curve_dict, field_shape, valid_range=20):
        # * estimate prior information
        prior_list = []
        curve_info = dict()
        for name, curve in curve_dict.items():
            curve_info[name] = []
            # length above win_k
            if len(curve) >= self.win_k:
                windows = np.lib.stride_tricks.sliding_window_view(np.arange(len(curve)), self.win_k)
                for window in windows:
                    X = curve[window, 0]
                    y = curve[window, 1]
                    if len(np.unique(X)) > 1:
                        _, k = linear_regression(X.reshape((-1, 1)), y)
                    else:
                        k = 0
                    prior_list.append([np.mean(X), np.mean(y), k])
                    curve_info[name].append(k)
            curve_info[name] = np.nanmean(curve_info[name])
        prior_info = np.array(prior_list)
        
        # * build parameter field
        h, w = field_shape
        prior_scatter = np.zeros(field_shape)
        for x_o, x_d, slope in prior_info:
            if x_o <= w and x_d <= h:
                prior_scatter[int(x_d-1), int(x_o-1)] = slope
        prior_field = np.zeros(field_shape)
        for x, y in np.ndindex(field_shape):
            x_start, y_start = max(0, x-valid_range), max(0, y-valid_range)
            search_range = prior_scatter[x_start:min(h, x+valid_range), y_start:min(w, y+valid_range)]
            valid_info = np.where(search_range != 0)
            if len(valid_info[0]) > 10:
                prior_slope = search_range[valid_info]
                prior_loc = np.array([valid_info[0]+x_start, valid_info[1]+y_start]).T
                w_vec = np.exp(-np.sum((prior_loc-np.array([x, y]))**2, axis=1)/(2*self.bw_para**2))
                w_vec = w_vec / np.sum(w_vec)
                prior_field[x, y] = np.sum(prior_slope * w_vec)
        return prior_field, curve_info

    def infer_posterior(self, curve_dict, prior_field, alpha=2):
        # * infer posterior 
        curve_new = dict()
        for name in curve_dict:
            curve = curve_dict[name]
            x_min, x_max = curve[:, 0].min(), curve[:, 0].max()
            x_vec = np.arange(int(x_min), int(x_max)+1)
            y_mean = curve[:, 1].mean()
            curve_hat = []
            for x in x_vec:
                slope_prior = prior_field[int(y_mean-1), int(x-1)]
                curve_hat.append([x, constrained_kernel_regression(x, curve[:, 0].reshape((-1, 1)), curve[:, 1], self.bw_data, slope_prior, 1, alpha).astype(np.int32)])
            curve_hat = np.array(curve_hat)
            curve_new[name] = curve_hat
        return curve_new
    
    
# 示例
if __name__ == "__main__":
    import numpy as np
    import cv2
    import os
    import matplotlib.pyplot as plt
    save_root = 'kernel_regression'
    source_dict = {
        'bp': '/home/htwang/result/cluster_seg_net/tuning_CBAM_1225/MSFSegNet-peak1-CBAM12-bs8-lr1e-03-BCE-Adam-S1/test-bp-AGC=5+9+15-10-3/samples/sample_Line00_Cdp3675.npy',
        'yz': '/home/htwang/result/cluster_seg_net/tuning_CBAM_1225/MSFSegNet-peak1-CBAM12-bs8-lr1e-03-BCE-Adam-S1/test-yz-AGC=11+31+51-10-3/samples/sample_Line18000_Cdp1880.npy',
        'jsh': '/home/htwang/result/cluster_seg_net/tuning_CBAM_1225/MSFSegNet-peak1-CBAM12-bs8-lr1e-03-BCE-Adam-S1/test-jsh-AGC=11+31+51-10-3/samples/sample_Line30_Cdp22060.npy',
        'sh': '/home/htwang/result/cluster_seg_net/tuning_CBAM_1225/MSFSegNet-peak1-CBAM12-bs8-lr1e-03-BCE-Adam-S1/test-sh-AGC=11+31+51-10-3/samples/sample_Line13800_Cdp4800.npy'
    }
    os.makedirs(save_root, exist_ok=True)
    for name, path in source_dict.items():
        res_dict = np.load(path, allow_pickle=True).item()
        curve_dict = res_dict['curve_auto']
        gth = res_dict['gth_msfeat'][0]

        infer_opt = posterior_regression(win_k=5, bw_data=5, bw_para=10)
        prior_field = infer_opt.est_prior(curve_dict, gth.shape)
        curve_dict_new = infer_opt.infer_posterior(curve_dict, prior_field)
        import sys
        sys.path.append('.')
        sys.path.append('..')
        sys.path.append('../..')
        from utils.plot_func import heatmap_plot
        heatmap_plot(gth, curve_dict=curve_dict, size=(3.5, 30), save_path=os.path.join(save_root, '%s-original.pdf'%name), cmap='binary')
        heatmap_plot(prior_field, size=(3.5, 30), save_path=os.path.join(save_root, '%s-field.pdf'%name), cmap='rainbow')
        heatmap_plot(gth, curve_dict=curve_dict_new, size=(3.5, 30), save_path=os.path.join(save_root, '%s-kernel.pdf'%name), cmap='binary')
    
    # 可视化
    # import matplotlib.pyplot as plt

    # plt.scatter(X, y, color='red', label='Data points')
    # plt.plot(x_values, predictions, color='blue', label='Normal Constrained Kernel Regression')
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Kernel Regression with Normal Constraint')
    # plt.savefig('local_linear_regression.png')
    
    
