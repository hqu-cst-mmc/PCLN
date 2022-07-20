import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
from config import *
from scipy import stats

env_path = './platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = env_path

class VideoDataset(Dataset):

    def __init__(self, mode, args):
        super(VideoDataset, self).__init__()
        # self.data_dir = data_dir
        self.mode = mode  # train or test
        # loading annotations
        self.args = args
        # self.annotations = pkl.load(open(os.path.join(info_dir, 'augmented_final_annotations_dict.pkl'), 'rb'))
        self.annotations = pkl.load(open(os.path.join(info_dir, 'augmented_norm_judge30_final_annotations_dict.pkl'), 'rb'))
        self.keys = pkl.load(open(os.path.join(info_dir, f'{self.mode}_split_0.pkl'), 'rb'))

    def get_fea(self, key):
        feature_list = os.path.join(feature_dir, str('{:02d}_{:02d}'.format(key[0], key[1])) + '.npy')       ##返回所有匹配的文件路径列表并排序
        feature_tensor = np.load(feature_list)
        return feature_tensor

    # def norm_judge(self, data):
    #     judge_score = data['judge_scores']
    #     judge_scores_sort = np.sort(judge_score)
    #     judge_label = np.sum(judge_scores_sort[2:5])
    #     # * data['difficulty'].cuda()
    #     # judge_scores_sort = torch.stack([judge_score], dim=1).sort()[0]# N, 7
    #     judge_min = judge_scores_sort[0]
    #     judge_max = judge_scores_sort[len(judge_scores_sort)-1]
    #     x_norm = (judge_label - judge_min) / (judge_max - judge_min)
    #     data['norm_judge_score'] = x_norm
    #     # return judge_label

    def proc_label(self, data):
        # Scores of MTL dataset ranges from 0 to 104.5, we normalize it into 0~100
        tmp = stats.norm.pdf(np.arange(101), loc=data['final_score'] * (101-1) / 104.5, scale=5).astype(np.float32)
        data['soft_label'] = tmp / tmp.sum()

        # Each judge choose a score from [0, 0.5, ..., 9.5, 10], we normalize the sum of judge scores into 0~30
        judge_scores_sort = np.sort(data['judge_scores'])
        judge_label = np.sum(judge_scores_sort[2:5])
        judge_tmp = stats.norm.pdf(np.arange(31), loc=judge_label * (31-1) / 29.0, scale=5).astype(np.float32)
        data['soft_judge_scores'] = judge_tmp / judge_tmp.sum()  # 7x21


    def __getitem__(self, ix):
        key = self.keys[ix]
        data = {}
        feature_list = os.path.join(feature_dir, str('{:02d}_{:02d}'.format(key[0], key[1])) + '.npy')
        feature_tensor = np.load(feature_list)
        # data['feature'] = self.get_fea(key)
        data['feature'] = feature_tensor
        data['final_score'] = self.annotations.get(key).get('final_score')
        data['norm_score'] = self.annotations.get(key).get('norm_score')       ##归一化后的总分
        data['difficulty'] = self.annotations.get(key).get('difficulty')
        data['judge_scores'] = self.annotations.get(key).get('judge_scores')
        data['norm_judge_scores'] = self.annotations.get(key).get('norm_judge_scores')
        data['norm30_judge_scores'] = self.annotations.get(key).get('norm30_judge_scores')
        self.proc_label(data)

        # #choose a sample
        # randomly
        file_list = self.keys.copy()
        # exclude self
        if len(file_list) > 1:
            file_list.pop(file_list.index(key))
        # choose one out
        idx_2 = random.randint(0, len(file_list) - 1)

        sample_2 = file_list[idx_2]
        target = {}
        target_feature_list = os.path.join(feature_dir, str('{:02d}_{:02d}'.format(sample_2[0], sample_2[1])) + '.npy')
        target_feature_tensor = np.load(target_feature_list)
        target['feature'] = target_feature_tensor
        target['final_score'] = self.annotations.get(sample_2).get('final_score')
        target['norm_score'] = self.annotations.get(sample_2).get('norm_score')  ##归一化后的总分
        target['difficulty'] = self.annotations.get(sample_2).get('difficulty')
        target['judge_scores'] = self.annotations.get(sample_2).get('judge_scores')
        target['norm_judge_scores'] = self.annotations.get(sample_2).get('norm_judge_scores')
        target['norm30_judge_scores'] = self.annotations.get(sample_2).get('norm30_judge_scores')

        return data, target

    def __len__(self):
        sample_pool = len(self.keys)
        return sample_pool
