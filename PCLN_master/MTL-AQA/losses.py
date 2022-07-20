import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from torch.autograd import Variable
from scipy.stats import spearmanr

# import cv2

# from visualize import make_dot
# import matplotlib.pyplot as plt   #

def gaussian_loss(output: torch.Tensor, label: torch.Tensor, sigma = 2) -> torch.Tensor:
    loss = torch.exp(- (output - label)**2 / (2 * sigma**2))
    # loss = torch.exp(- (output - label) ** 2 / (5 * sigma**2))

    return loss

class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, predict, label):
        loss = gaussian_loss(predict, label)
        return 1.0 - loss


def get_rank(x: torch.Tensor) -> torch.Tensor:
    x = x * 10000
    int_x = x.int().flatten()
    tmp = int_x.argsort()
    # rank = torch.zeros_like(tmp)
    size = int_x.shape[0]
    ranks = torch.empty(size)
    for i in range(size):
        ranks[tmp[i]] = i
    # ranks[tmp] = torch.arange(len(int_x))
    return ranks

def spearman_loss(x: torch.Tensor, y: torch.Tensor):
    x_rank = get_rank(x)
    y_rank = get_rank(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n ** 2 - 1.0)
    # return 1- (upper / down)
    return 1.0 - (upper / down)

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, truth):
        res = spearman_loss(pred, truth)
        # return 0.01 * (1.0 - res)
        return 1.0 - res
        # rho, p_val = spearmanr(pred.data.cpu().numpy(), truth.data.cpu().numpy())
        # rho, p_val = spearmanr(pred, truth)
        # return 1.0 - rho
