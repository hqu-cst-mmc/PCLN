from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import repeat
import math
from functools import partial


class my_fc(nn.Module):
    def __init__(self):
        super(my_fc, self).__init__()
        self.fc1 = nn.Linear(40 * 512, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, h):
        h = h.view(-1, 20480)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.fc3(h)
        # h = self.relu(self.fc3(h))
        # h = self.fc4(h)
        h = self.sig(h)
        return h
