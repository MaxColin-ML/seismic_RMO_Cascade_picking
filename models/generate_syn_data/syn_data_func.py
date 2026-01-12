import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


###### ricker子波：从左到右，频率需要从高渐变到低 ######
def RickerWavelet(t, dr=0, freq=10):
    t = np.arange(t) #生成数组
    exp = (np.pi * (t-dr) * freq / 1000) ** 2
    return (1 - 2 * exp) * np.exp(-exp)


###### z^2=z0^2+beta*h^2+gamma*h^4 #######
# beta > 0:双曲线 下垂 | beta < 0:椭圆 上扬
def pick(w_len, z0, beta, gamma=0):
    # 生成一个长度为 w_len 的数组，包含从 0 到 w_len-1 的数值
    h = np.arange(w_len)
    # 计算 z^2 的值
    # 把曲率整体向左移5个单位
    z_square = z0**2 + beta * (h+5)**2 + gamma * (h+5)**4
    # 计算 z
    z = np.sqrt(z_square)  
    return z


# Generate the gather data according the t-v set
def synthetic_data(k, h, w, beta_range_neg, beta_range_pos, gamma_range_neg, gamma_range_pos, freq_range, add_angle_noise=False):
    """
    生成合成地震数据
    :param k: 曲率的个数
    :param h: 数据的高度
    :param w: 数据的宽度
    :param beta_range_neg: 负向二次项系数范围
    :param beta_range_pos: 正向二次项系数范围
    :param gamma_range_neg: 负向四次项系数范围
    :param gamma_range_pos: 正向四次项系数范围
    :param freq_range: 主频范围，形如 (min_freq, max_freq)
    :return: 合成地震数据数组，大小为 (h, w)
    """
    t_o_list_pos = []
    pick_dict = {}  # 保存pick曲线坐标的字典

    # compute the interval
    start_t = np.random.randint(10, 200)
    start_col = np.random.randint(20, 50)
    incre_rate = np.random.uniform(0.035, 0.055)
    samp_itv = 5
    curve_interval = ((h-start_t) / samp_itv) / k 
    # 正向数据生成
    for i in range(k):
        interval_noise = np.random.uniform(-0.0005, 0.0005)
        z0 = start_t/samp_itv + i*(curve_interval+interval_noise)
        if i <= int(k/3):
            beta = round(np.random.uniform(beta_range_pos[0], beta_range_pos[1]), 3)
            gamma = round(np.random.uniform(gamma_range_pos[0], gamma_range_pos[1]), 6)
        elif i <= int(k/2):
            beta = round(np.random.uniform(0.005, 0.02), 3)
            gamma = round(np.random.uniform(0.000008, 0.000015), 6)
        elif i <= int(2*k/3):
            beta = round(np.random.uniform(-0.02, -0.005), 3)
            gamma = round(np.random.uniform(-0.000010, -0.000005), 6)
        else:
            beta = round(np.random.uniform(beta_range_neg[0], beta_range_neg[1]), 3)
            gamma = round(np.random.uniform(gamma_range_neg[0], gamma_range_neg[1]), 6)

        pick_value1 = pick(w, z0, beta, gamma)
        # whether beyond the boundary
        if np.sum((pick_value1*samp_itv) >= h) > 0:
            continue
        curve_crop = [[j+1, int(t_s*samp_itv)+1] for j, t_s in enumerate(pick_value1) if j < int(t_s*samp_itv*incre_rate+start_col)]
        pick_dict[f'curve_pos{i+1}'] = curve_crop
        for j, t_s in curve_crop:
            t_o_list_pos.append([t_s-1, j-1, i])

    # 初始化正向合成地震数据数组
    synthetic_data_pos = np.zeros((h, w))
    freq_step = (freq_range[1] - freq_range[0]) / k
    freq = np.zeros(k)
    freq_noise = 5
    scale = np.zeros(k)
    scale_noise = 5
    weak_percent = 0.4
    weak_rate = np.linspace(1, 1-weak_percent, w)
    for i in range(k):
        freq[i] = freq_range[1] - freq_step * i + np.random.uniform(-freq_noise, freq_noise)
        scale_k = np.random.uniform(35, 50)
        scale[i] = scale_k + np.random.uniform(-scale_noise, scale_noise)
        
    for z, trace, i in t_o_list_pos:
        synthetic_data_pos[:, trace] += scale[i] * RickerWavelet(h, z, freq[i]*weak_rate[trace])
    for i in range(0, h):
        j = int(min(start_col+incre_rate*i, w))
        synthetic_data_pos[i, j:] = 0
        
    synthetic_data = synthetic_data_pos
    
    # 生成角噪声
    if add_angle_noise:
        # control parameters
        half_wid = np.random.randint(2, 4)
        h1 = np.random.randint(3, 10)
        w1, w2 = np.random.randint(3, 15), np.random.randint(20, 60)
        h_angle_noise = np.arange(h1, h)
        w_angle_noise = (h_angle_noise-h1) * (w2-w1) / (h-h1) + w1
        w_angle_noise = w_angle_noise.astype(int)
        angle_noise_points = np.array([h_angle_noise, w_angle_noise]).T
        for loc_h, loc_w in angle_noise_points:
            synthetic_data[loc_h, max(loc_w-half_wid, 0):min(loc_w+half_wid+1, w)] = np.random.uniform(0, 0.2) * np.max(synthetic_data_pos)
    
    return synthetic_data, pick_dict


def add_gaussian_noise(synthetic_data, scale):
    """
    给合成地震数据添加高斯噪声
    
    :param synthetic_data: 合成地震数据数组，大小为 (h, w)
    :param scale: 高斯噪声的尺度
    :return: 添加了高斯噪声的合成地震数据数组
    """
    # 生成高斯噪声
    # scale使高斯分布的标准差，越小分布越极端
    noise = np.random.normal(scale=scale, size=synthetic_data.shape)
    # 添加噪声到合成数据上
    noisy_data = synthetic_data + 10*noise
    return noisy_data
