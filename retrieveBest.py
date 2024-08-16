import torch
import torch.nn as nn
import torch.nn.functional as F

from myAttention import *

class HybridSN(nn.Module):
    def __init__(self, rate=16, class_num=2, windowSize=25, K=30):
        super(HybridSN, self).__init__()
        self.S = windowSize
        self.L = K

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3))
        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        
        inputX = self.get2Dinput()
        inputConv4 = inputX.shape[1] * inputX.shape[2]
        self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3))

        self.sa1 = nn.Conv2d(64, 64//rate, kernel_size=1)
        self.sa2 = nn.Conv2d(64//rate, 64, kernel_size=1)
        
        self.dense1 = nn.Linear(4096*4, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, class_num)

        self.senet = FusionAttenion(channel=4, ratio=4)
        self.senet2 = SeNet(channel=204, ratio=1/rate)
        self.spatial_attention1 = spacial_attention()
        self.spatial_attention2 = spacial_attention()
        self.spatial_attention3 = spacial_attention()
        self.spatial_attention4 = spacial_attention()

        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(1)

        self.fc11 = nn.Linear(K, 1024)
        self.fc21 = nn.Linear(1024, 2048)
        self.fc31 = nn.Linear(2048, 4096)

        self.fc12 = nn.Linear(K, 1024)
        self.fc22 = nn.Linear(1024, 2048)
        self.fc32 = nn.Linear(2048, 4096)

        self.fc13 = nn.Linear(K, 1024)
        self.fc23 = nn.Linear(1024, 2048)
        self.fc33 = nn.Linear(2048, 4096)

        pass

    def get2Dinput(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.L, self.S, self.S))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x
    
    def getCharacter(self, x):
        if True:
            x = x.view([x.shape[0], x.shape[2], x.shape[3], x.shape[4]])
            x = self.spatial_attention1(x)
            x = self.senet2(x)
            x = x.view([x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]])

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
        out = F.relu(self.conv4(out))
        return out
    
    def getCharacterSoilUp(self, x):
        if True:
            x = x.view([x.shape[0], x.shape[2], x.shape[3], x.shape[4]])
            x = self.spatial_attention2(x)
            x = x.view([x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]])
        x = self.avg_pool2(x)
        x = x.view([x.shape[0], -1])
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc21(x))
        x = F.relu(self.fc31(x))
        return x
    
    def getCharacterSoilDown(self, x):
        if True:
            x = x.view([x.shape[0], x.shape[2], x.shape[3], x.shape[4]])
            x = self.spatial_attention3(x)
            x = x.view([x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]])
        x = self.avg_pool3(x)
        x = x.view([x.shape[0], -1])
        x = F.relu(self.fc12(x))
        x = F.relu(self.fc22(x))
        x = F.relu(self.fc32(x))
        return x
    
    def getCharacterStem(self, x):
        if True:
            x = x.view([x.shape[0], x.shape[2], x.shape[3], x.shape[4]])
            x = self.spatial_attention4(x)
            x = x.view([x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]])
        x = self.avg_pool4(x)
        x = x.view([x.shape[0], -1])
        x = F.relu(self.fc13(x))
        x = F.relu(self.fc23(x))
        x = F.relu(self.fc33(x))
        return x

    def forward(self, x, img_soilUp, img_soilDwon, img_stem):
        out = self.getCharacter(x)
        out_soilUp = self.getCharacterSoilUp(img_soilUp)
        out_soilDown = self.getCharacterSoilDown(img_soilDwon)
        out_stem = self.getCharacterStem(img_stem)

        out = out.view([out.size(0), 1, -1])
        out_soilUp = out_soilUp.view([out_soilUp.size(0), 1, -1])
        out_soilDown = out_soilDown.view([out_soilDown.size(0), 1, -1])
        out_stem = out_stem.view([out_stem.size(0), 1, -1])

        out = torch.cat([out, out_soilUp, out_soilDown, out_stem], 1)
        
        if True:
            out = self.senet(out)
        out = out.view([out.size(0), -1])

        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)

        return out