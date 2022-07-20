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


class decoder_edtcn(nn.Module):

    def __init__(self):
        super(decoder_edtcn, self).__init__()
        ## ? * 160 * 2048

        self.upsample1 = nn.Upsample(80)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=1)
        self.drop3 = SpatialDropout(0.3)
        self.upsample2 = nn.Upsample(160)
        self.conv4 = nn.Conv1d(512, 256, kernel_size=1)
        self.drop4 = SpatialDropout(0.3)

        self.fc = nn.Linear(256, 5)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, h):
        h = self.upsample1(h)
        h = self.conv3(h)
        h = self.drop3(h)
        h = self.relu(h)

        h = self.upsample2(h)
        h = self.conv4(h)
        h = self.drop4(h)
        h = self.relu(h)
        h = h.permute(0, 2, 1)
        h = self.softmax(self.fc(h))

        return h




