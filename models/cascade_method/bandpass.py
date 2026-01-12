##########################################################
# Apply bandpass filter on a 2D gather 
# ---
# Author: Chatgpt (revised by Hongtao Wang)
##########################################################

import torch.fft


def bandpass_filter_2d(input_tensor, low_cutoff=1, high_cutoff=5, fs=10):
    """
    对 4D 输入张量 (bs, 1, h, w) 在 h 维度（高度方向）进行带通滤波
    :param input_tensor: 输入张量，形状为 (bs, 1, h, w)
    :param low_cutoff: 低截止频率
    :param high_cutoff: 高截止频率
    :param fs: 采样频率
    :return: 滤波后的张量
    """
    # 获取输入张量的形状
    bs, _, h, w = input_tensor.shape

    # 创建一个空的输出张量
    output_tensor = torch.zeros_like(input_tensor)

    # 对每个样本进行处理
    for i in range(bs):
        # 获取单个样本，形状为 (1, h, w)
        sample = input_tensor[i, 0, :, :]
        for j in range(w):
            # 执行 FFT 变换，得到频域表示
            fft_sample = torch.fft.fft(sample[:, j])
            # 频率轴
            freqs = torch.fft.fftfreq(h, d=1/fs)
            # 构造带通滤波器
            bandpass_filter = torch.zeros_like(fft_sample)
            bandpass_filter[(freqs >= low_cutoff) & (freqs <= high_cutoff)] = 1
            # 应用带通滤波器
            filtered_fft_sample = fft_sample * bandpass_filter
            # 执行逆 FFT 变换，得到时域信号
            filtered_sample = torch.fft.ifft(filtered_fft_sample)
            # 将滤波后的样本存回输出张量
            output_tensor[i, 0, :, j] = filtered_sample.real

    return output_tensor