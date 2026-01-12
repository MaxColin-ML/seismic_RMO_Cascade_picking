import numpy as np


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


def calculate_slope_field(curve_dict, field_shape, win_k=5, bw_para=5, valid_range=50):
    # 以上超参数是默认参数
    
    # * estimate prior information
    prior_list = []
    curve_info = dict()
    for name, curve in curve_dict.items():
        curve_info[name] = []
        # length above win_k
        if len(curve) >= win_k:
            windows = np.lib.stride_tricks.sliding_window_view(np.arange(len(curve)), win_k)
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
            w_vec = np.exp(-np.sum((prior_loc-np.array([x, y]))**2, axis=1)/(2*bw_para**2))
            w_vec = w_vec / np.sum(w_vec)
            prior_field[x, y] = np.sum(prior_slope * w_vec)
            
    return prior_field