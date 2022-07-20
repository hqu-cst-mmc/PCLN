import torch.nn as nn
from config import *
from models_for_ED_TCN.encoder_model import *

class Evaluator(nn.Module):

    def __init__(self, model_type='single', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'single':
            self.evaluator = encoder_edtcn()
        else:
            assert num_judges is not None, 'num_judges is required in Multi'
            self.evaluator = nn.ModuleList([encoder_edtcn() for _ in range(num_judges)])

    def forward(self, feats_avg):

        if self.model_type == 'single':
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs
