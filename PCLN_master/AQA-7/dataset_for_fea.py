import os
from os import listdir
from os.path import isfile, join, isdir

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import random
from PIL import Image


class divingDataset(Dataset):
    def __init__(self, data_folder, data_file, label_file, spss_file, test=0):

        self.data_folder = data_folder
        ## data_file: train/test
        self.video_name = np.load(data_file)
        self.label = np.load(label_file)
        self.test = test
        # self.diff = np.load(difficulty_file)
        self.spss = np.load(spss_file)

    def __getitem__(self, index):

        video_name = str(self.video_name[index][0]).zfill(3)

        feature_path = os.path.join(self.data_folder, video_name+'.npy')
        video_tensor = np.load(feature_path)
        labels = self.label[0][self.video_name[index][0] - 1].astype(np.float32)
        # diffs = self.diff[0][self.video_name[index][0] - 1].astype(np.float32)
        spss_labels = self.spss[0][self.video_name[index][0] - 1].astype(np.float32)

        return video_tensor, labels, spss_labels


    def __len__(self):
        return len(self.video_name)
