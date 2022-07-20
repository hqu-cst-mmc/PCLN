import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
# import cv2
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models_for_ED_TCN.encoder_model import encoder_edtcn
from models_for_ED_TCN.score_regressor import Evaluator

from dataset_for_mtl import VideoDataset
# from visualize import make_dot
from scipy.stats import spearmanr
from tqdm import tqdm
# import matplotlib.pyplot as plt   #
from plot_for_vis import plot_loss_spss
from losses import MyLoss
from losses import *
from config import *
from config import get_parser
from utils import *

def compute_loss(options, output, label, data):
    if options.type == 'single':
        if options.loss == "MSELoss":
            criterion = nn.MSELoss()  ## MSEloss
            loss = criterion(output, label)
        elif options.loss == "MSELoss+L1Loss":
            criterion_final_score = nn.MSELoss()
            penalty_final_score = nn.L1Loss()
            loss = (criterion_final_score(output, label) + penalty_final_score(output, label))
        elif options.loss == "MyLoss+MSE":
            criterion_score = MyLoss()
            criterion_final_score = nn.MSELoss()
            loss = (0.01 * criterion_score(output, label) + criterion_final_score(output, label))
        elif options.loss == "MyLoss+MSE+L1Loss":
            criterion_final_score = nn.MSELoss()
            penalty_final_score = nn.L1Loss()
            criterion_score = MyLoss()
            loss = (criterion_score(output, label) + criterion_final_score(output, label)
                    + penalty_final_score(output, label))
        elif options.loss == "SmoothL1loss":
            criterion = nn.SmoothL1Loss()
            loss = criterion(output, label)
        elif options.loss == "Gaussian_loss":
            criterion = GaussianLoss()
            loss = criterion(output, label)
            loss = torch.sum(loss) / options.train_batch_size
        elif options.loss == "Gaussian+Myloss":
            criterion_gau = GaussianLoss()
            criterion_my = MyLoss()
            loss_gau = criterion_gau(output, label)
            loss_gau = torch.sum(loss_gau) / options.train_batch_size
            loss_my = criterion_my(output, label)
            loss = loss_gau + 0.00001 * loss_my
    else:
        if options.loss == "MSELoss":
            criterion = nn.MSELoss()  ## MSEloss
            loss = sum([criterion(output[i], data['judge_scores'][:, i].float().cuda()) for i in range(7)])

        elif options.loss == "MSELoss+L1Loss":
            criterion_final_score = nn.MSELoss()
            penalty_final_score = nn.L1Loss()
            loss = (criterion_final_score(output, label) + penalty_final_score(output, label))
        elif options.loss == "MyLoss+MSE":
            criterion_score = MyLoss()
            criterion_final_score = nn.MSELoss()
            loss = (0.01 * criterion_score(output, label) + criterion_final_score(output, label))
        elif options.loss == "MyLoss+MSE+L1Loss":
            criterion_final_score = nn.MSELoss()
            penalty_final_score = nn.L1Loss()
            criterion_score = MyLoss()
            loss = (criterion_score(output, label) + criterion_final_score(output, label)
                    + penalty_final_score(output, label))
        elif options.loss == "SmoothL1loss":
            criterion = nn.SmoothL1Loss()
            loss = criterion(output, label)
        elif options.loss == "Gaussian_loss":
            criterion = GaussianLoss()
            # loss = [criterion(output[i], data['judge_scores'][:, i].cuda()) for i in range(7)]
            # loss = sum(loss)
            loss = sum([criterion(output[i], data['judge_scores'][:, i].cuda() / 10) for i in range(7)])
            loss = torch.sum(loss) / options.train_batch_size

        elif options.loss == "Gaussian+Myloss":
            criterion_gau = GaussianLoss()
            criterion_my = MyLoss()
            # loss_gau = criterion_gau(output, label)
            loss_gau = sum([criterion_gau(output[i], data['judge_scores'][:, i].cuda()) for i in range(7)])
            loss_gau = torch.sum(loss_gau) / options.train_batch_size
            loss_my = sum([criterion_my(output[i], data['judge_scores'][:, i].cuda()) for i in range(7)])
            loss = loss_gau + 0.00001 * loss_my
        elif options.loss == "Gaussian+MSELoss":
            criterion_gau = GaussianLoss()
            criterion_mse = nn.MSELoss()
            loss_gau = sum([criterion_gau(output[i], data['judge_scores'][:, i].cuda() / 10) for i in range(7)])
            loss_gau = torch.sum(loss_gau) / options.train_batch_size
            loss_mse = sum([criterion_mse(output[i], data['judge_scores'][:, i].float().cuda() / 10) for i in range(7)])
            loss = loss_gau + loss_mse

    return loss
