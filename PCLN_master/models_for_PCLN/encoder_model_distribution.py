from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import repeat
import math
from functools import partial

class SpatialDropout(nn.Module):
    def __init__(self, drop):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])  #默认沿着中间所有的shape

        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)

class Lambda(nn.Module):
    # Normalize by the highest activation
    def __init__(self, channel_normalization):
        super(Lambda, self).__init__()
        self.channel_normalization = channel_normalization

    def forward(self, x):
        return  self.channel_normalization(x)


class encoder_edtcn(nn.Module):

    def __init__(self):
        super(encoder_edtcn, self).__init__()
        ## ? * 160 * 2048

        self.conv1 = nn.Conv1d(2048, 256, kernel_size=1)
        self.drop1 = SpatialDropout(0.3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(256, 512, kernel_size=1)
        self.drop2 = SpatialDropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(40 * 512, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 101)
        # self.fc3 = nn.Linear(2048, 31)
        # self.fc4 = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x_t = x.permute(0, 2, 1)
        h = x_t.to(torch.float32)
        h = self.conv1(h)
        h = self.drop1(h)
        h = self.relu(h)
        # h = self.channel_norm1(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.drop2(h)
        h = self.relu(h)
        # h = self.channel_norm2(h)
        h = self.pool2(h)

        h = h.view(-1, 20480)
        # h = self.fc1(h)
        # h = self.fc2(h)
        # h = self.fc3(h)
        # h = self.fc4(h)

        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        output = self.softmax(self.fc3(h))
        return output




# def EDTCN_encoder():
#
#     model = ED_TCN()
#     return model
#
# if __name__ == '__main__':
#     EDTCN_encoder()
#

