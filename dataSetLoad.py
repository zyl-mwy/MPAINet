import spectral 
import pandas as pd
from torch.utils.data import Dataset
import os
import torch

class Hyper_IMG(Dataset):

    def __init__(self, root, train='Train', transform = None, target_transform=None):
        super(Hyper_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        img_folder = root + '/data/leaf/maskPicNormal/enhance/'
        img_soil_folder = root + '/data/soil/maskPicNormal/enhance/'
        img_stem_folder = root + '/data/stem/maskPicNormal/enhance/'

        if self.train=='Train':
            file_annotation = root + '/data/dataInfo/edition/Final_NewEnhanceTrain.xlsx'
        elif self.train=='Val':
            file_annotation = root + '/data/dataInfo/edition/Final_NewEnhanceVal.xlsx'
        elif self.train=='Test':
            file_annotation = root + '/data/dataInfo/edition/Final_NewEnhanceTest.xlsx'
        myAnnotion = pd.read_excel(file_annotation).values
        self.num_data = myAnnotion.shape[0]
        self.filenames = []
        self.labels1 = []
        self.labels2 = []
        self.soilUp = []
        self.soilDown = []
        self.stem = []
        self.rootWhole = []
        self.img_folder = img_folder
        self.img_soil_folder = img_soil_folder
        self.img_stem_folder = img_stem_folder
        for i in range(self.num_data):
            self.filenames.append(myAnnotion[i][2])
            self.labels1.append(myAnnotion[i][8])
            self.labels2.append(myAnnotion[i][9])
            self.soilUp.append(myAnnotion[i][3])
            self.soilDown.append(myAnnotion[i][4])
            self.stem.append(myAnnotion[i][10])
            self.rootWhole.append(myAnnotion[i][11])

    def __getitem__(self, index):
        img_name = os.path.join(self.img_folder, self.filenames[index])
        img_soilUp_name = os.path.join(self.img_soil_folder, self.soilUp[index])
        img_soilDown_name = os.path.join(self.img_soil_folder, self.soilDown[index])
        img_stem_name = os.path.join(self.img_stem_folder, self.stem[index])
        label = torch.FloatTensor([self.labels1[index] / 500, self.labels2[index] / 500])
        img = (spectral.envi.open(img_name+'.hdr', img_name+'.img').read_bands([i for i in range(204)])-0.58)/0.39
        img_soilUp = (spectral.envi.open(img_soilUp_name+'.hdr', img_soilUp_name+'.img').read_bands([i for i in range(204)])-0.58)/0.39
        img_soilDwon = (spectral.envi.open(img_soilDown_name+'.hdr', img_soilDown_name+'.img').read_bands([i for i in range(204)])-0.58)/0.39
        img_stem = (spectral.envi.open(img_stem_name+'.hdr', img_stem_name+'.img').read_bands([i for i in range(204)])-0.58)/0.39
        
        if self.transform is not None:
            img = torch.unsqueeze(self.transform(img).permute(1, 2, 0), dim=0)
            img_soilUp = torch.unsqueeze(self.transform(img_soilUp).permute(1, 2, 0), dim=0)
            img_soilDwon = torch.unsqueeze(self.transform(img_soilDwon).permute(1, 2, 0), dim=0)
            img_stem = torch.unsqueeze(self.transform(img_stem).permute(1, 2, 0), dim=0)

        return img, img_soilUp, img_soilDwon, img_stem, label

    def __len__(self):
        return self.num_data