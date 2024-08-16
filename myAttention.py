import torch
from torch import nn

class SeNet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeNet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*ratio), False),
            nn.ReLU(),
            nn.Linear(int(channel*ratio), channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])
        return x*fc
    
class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out*x
    
class FusionAttenion(nn.Module):
    def __init__(self, channel, ratio=16):
        super(FusionAttenion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*ratio), False),
            nn.ReLU(),
            nn.Linear(int(channel*ratio), channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1])
        return x*fc