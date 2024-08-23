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
    

class self_attetion(nn.Module):
    def __init__(self, channel, ratio, size):
        super(self_attetion, self).__init__()
        self.avg_pool = nn.AvgPool2d(size)
        self.sa1 = nn.Conv2d(channel, channel//ratio, kernel_size=1)
        self.sa2 = nn.Conv2d(channel//ratio, channel, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.relu(self.sa1(weight))
        weight = self.sigmoid(self.sa2(weight))
        out = x * weight
        return out
    