import torch
from torch import nn
class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()
    def forward(self, y_pred, y_true):
        y_mean = torch.mean(y_true)
        ss_lot = torch.sum((y_true - y_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        return ss_res / ss_lot