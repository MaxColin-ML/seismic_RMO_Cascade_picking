##########################################################
# Curvature segmentation networks
# U-Net and CBAM module
# ---
# Author: Hongtao Wang & ChatGPT
# Email: colin315wht@gmail.com
##########################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from models.cascade_method.DSConv import DSConv_layer
from models.cascade_method.bandpass import bandpass_filter_2d
"""
This file defines a U-Net model.
U-Net 
---
upsamling method = nn.ConvTranspose2d

2024/7/17 在网络前加入AGC

"""
class AGC_Torch(nn.Module):
    def __init__(self, AGC_len=11, device = 1):
        # AGC_len=31：设置AGC的窗口长度为31
        # device=0：默认设备是0，表示使用GPU设备0
        # self.avg_kernel：创建一个大小为(1, 1, AGC_len, 1)的平均滤波器，值全部为1。这个滤波器将用于计算局部平均值
        super(AGC_Torch, self).__init__()
        self.avg_kernel = torch.ones((1, 1, AGC_len, 1), device = device)/AGC_len
        self.pad = nn.ZeroPad2d(padding=(0, 0, (AGC_len-1)//2, (AGC_len-1)//2))
    
    def forward(self, trace):  # input: shape = (bs, 1, H, W)
        trace_abs = torch.square(trace)
        trace_abs = self.pad(trace_abs)
        avg_map = torch.sqrt(F.conv2d(trace_abs, self.avg_kernel, stride=(1, 1)))
        agc_map = trace/(avg_map+1e-8)
        return agc_map
 
 
def AGC_numpy(gth, AGC_len=11):
    agc_opt = AGC_Torch(AGC_len=AGC_len, device='cpu')
    # transform to torch tensor
    gth = torch.from_numpy(gth).float().unsqueeze(0).unsqueeze(0)
    gth_agc = agc_opt(gth)
    # transform to numpy
    gth_agc = gth_agc.squeeze().cpu().numpy()
    return gth_agc
     
     
def sqrt_norm(gth):
    gth_sqrt = torch.sqrt(torch.mean(gth**2, dim=2, keepdim=True))
    gth_norm = gth / (gth_sqrt+1e-10)
    return gth_norm

def tw_norm(gth):
    gth_max, _ = torch.max(torch.abs(gth), dim=2, keepdim=True)
    gth_norm = gth / (gth_max+1e-10)
    return gth_norm
    
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
 
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
 
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
 
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# Down Sampling Module
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', down_sample=True):
        super(UNetDownBlock, self).__init__()
        layers = []
        if down_sample:
            layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True))
        layers.append(nn.BatchNorm2d(out_channels))
        if act == 'elu':
            layers.append(nn.ELU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'tanh':
            layers.append(nn.Tanh())
        else:
            raise NotImplementedError
        layers.append(nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True))
        layers.append(nn.BatchNorm2d(out_channels))
        if act == 'elu':
            layers.append(nn.ELU(inplace=True))
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'tanh':
            layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Up Sampling Module
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.ReLU()
        )

    def forward(self, x, skip_input):
        x = self.up_sample(x)
        x = torch.cat((x, skip_input), 1)#沿着通道维拼接
        x = self.conv(x)
        return x


class FirstDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, extend_scope=1.0, basic_act='relu', device=0):
        super(FirstDownBlock, self).__init__()
        self.conv = UNetDownBlock(in_channels, out_channels, act=basic_act, down_sample=False)
        self.convx = DSConv_layer(in_channels, out_channels, kernel_size=kernel_size, extend_scope=extend_scope,morph=0 , if_offset=True, device=device).to(device)
        self.conv_fuse = UNetDownBlock(2 * out_channels, out_channels, act='relu', down_sample=False)

    def forward(self, X):
        x1_1 = self.conv(X)
        x2_1 = self.convx(X)
        out = self.conv_fuse(torch.cat([x1_1, x2_1], dim=1))
        return out
    

class MSFSegNet(nn.Module):
    def __init__(self, agc_list=[31, 51], bp_list=[1, 20, 20], CBAM_reduction=16, basic_act='tanh', dcn_use=1, cbam_use=1, add_peak=1, add_bp=1, device=0):
        super(MSFSegNet, self).__init__()
        self.agc_opts = [AGC_Torch(AGC_len=agc, device=device) for agc in agc_list]
        self.bp_list = bp_list
        if dcn_use:
            self.down1 = FirstDownBlock(len(agc_list)+add_peak+add_bp, 32, kernel_size=9, extend_scope=1.0, basic_act=basic_act, device=device)
        else:
            self.down1 = UNetDownBlock(len(agc_list)+add_peak+add_bp, 32, act=basic_act, down_sample=False)
        if cbam_use:
            self.CBAM = CBAMLayer(32, reduction=CBAM_reduction, spatial_kernel=9)
        self.down2 = UNetDownBlock(32, 64)
        self.down3 = UNetDownBlock(64, 128)
        self.center = UNetDownBlock(128, 256)
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)
        self.last = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.weight_init()
        self.add_peak = add_peak
        self.add_bp = add_bp
        self.cbam_use = cbam_use
        
    def forward(self, x):
        x_ori = x[:, 0, :, :].unsqueeze(1)
        x_peak = x[:, 1, :, :].unsqueeze(1)
        
        # feature extraction
        x_ms_list = []
        x_agc = torch.concat([opt(x_ori) for opt in self.agc_opts], dim=1)  
        x_ms_list.append(x_agc)
        if self.add_bp:
            x_bf = bandpass_filter_2d(x_agc[:, -1].unsqueeze(1), low_cutoff=self.bp_list[0], high_cutoff=self.bp_list[1], fs=self.bp_list[2]) 
            x_bf[torch.abs(x_agc[:, -1].unsqueeze(1))<0.1] = 0
            x_ms_list.append(x_bf)
        if self.add_peak:
            x_ms_list.append(x_peak)
        x_msfeat = torch.concat(x_ms_list, dim=1)
        x_norm = tw_norm(x_msfeat)
        d1 = self.down1(x_norm)  # 32     H       W
        if self.cbam_use:
            d1 = self.CBAM(d1)
        d2 = self.down2(d1)  # 64    H/2     W/2
        d3 = self.down3(d2)  # 128    H/4     W/4
        out = self.center(d3)  # 256  H/8   W/8
        out = self.up3(out, d3)  # 128    H/4     W/4
        out = self.up2(out, d2)  # 64    H/2     W/2
        out = self.up1(out, d1)  # 32     H       W
        out = self.last(out)  # 1     H       W
        return out.squeeze(), x_norm
    
    def weight_init(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0) 
            
    def change_agc(self, new_agc_list, device):
        self.agc_opts = [AGC_Torch(AGC_len=agc, device=device) for agc in new_agc_list]


class UNet(nn.Module):
    def __init__(self, agc_list=[31, 41, 51], add_peak=1, device=0):
        super(UNet, self).__init__()
        self.agc_opts = [AGC_Torch(AGC_len=agc, device=device) for agc in agc_list]
        self.down1 = UNetDownBlock(len(agc_list)+add_peak+1, 32, False)
        self.down2 = UNetDownBlock(32, 64)
        self.down3 = UNetDownBlock(64, 128)
        self.center = UNetDownBlock(128, 256)
        self.up3 = UNetUpBlock(256, 128)
        self.up2 = UNetUpBlock(128, 64)
        self.up1 = UNetUpBlock(64, 32)
        self.last = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.weight_init()
        self.add_peak = add_peak

    def sqrt_norm(self, gth):
        gth_sqrt = torch.sqrt(torch.sum(gth**2, dim=2, keepdim=True))
        gth_norm = gth / (gth_sqrt+1e-10)
        return gth_norm
    
    def forward(self, x):
        x_ori = x[:, 0, :, :].unsqueeze(1)
        x_peak = x[:, 1, :, :].unsqueeze(1)
        x_norm = self.sqrt_norm(x_ori)
        # 在网络前进行AGC处理
        x_agc_norm = torch.concat([self.sqrt_norm(opt(x_ori)) for opt in self.agc_opts], dim=1) 
        if self.add_peak:
            x_concat = torch.concat((x_norm, x_peak, x_agc_norm), dim=1)
        else:
            x_concat = torch.concat((x_norm, x_agc_norm), dim=1)
        d1 = self.down1(x_concat)  # 32     H       W
        d2 = self.down2(d1)  # 64    H/2     W/2
        d3 = self.down3(d2)  # 128    H/4     W/4
        out = self.center(d3)  # 256  H/8   W/8
        out = self.up3(out, d3)  # 128    H/4     W/4
        out = self.up2(out, d2)  # 64    H/2     W/2
        out = self.up1(out, d1)  # 32     H       W
        out = self.last(out)  # 1     H       W
        return out.squeeze(), x_concat

    def weight_init(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, -100)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0) 


if __name__ == "__main__":
    #x_test = torch.rand(size=(2, 1, 256, 128), device=1)
    x_test = torch.rand(size=(16, 2, 256+8, 256+8), device=1)
    net = MSFSegNet(device=1)
    net.cuda(1)
    out_1, x = net(x_test)
    print(out_1.size())
    print("ok")
    x_bf = bandpass_filter_2d(x_test, low_cutoff=10, high_cutoff=200, fs=30)
    print(x_bf.shape)
