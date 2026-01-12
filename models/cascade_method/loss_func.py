##########################################################
# Mixed loss function
# BCE loss and texture loss
# ---
# Author: Hongtao Wang
# Email: colin315wht@gmail.com
##########################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class texture_loss(nn.Module):
    def __init__(self, device):
        super(texture_loss, self).__init__()
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        pred_4d = pred.unsqueeze(1)
        target_4d = target.unsqueeze(1)
        pred_edge_x = F.conv2d(pred_4d, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_4d, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target_4d, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_4d, self.sobel_y, padding=1)

        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2)

        return F.mse_loss(pred_edge, target_edge)
    
    
class mix_loss(nn.Module):
    def __init__(self, device):
        super(mix_loss, self).__init__()
        self.loss_tt = texture_loss(device)
        self.loss_bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.loss_bce(pred, target)
        tt_loss = self.loss_tt(pred, target)
        a = (bce_loss/(tt_loss+1e-10)).detach()
        loss = bce_loss + a * tt_loss 
        return loss