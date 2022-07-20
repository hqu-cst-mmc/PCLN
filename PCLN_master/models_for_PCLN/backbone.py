from models_for_ED_TCN.encoder_model_contrastive import encoder_edtcn
from models_for_ED_TCN.my_fc import my_fc
import torch.nn as nn
import torch

def compute_group(input_fea):
    group_feature = torch.empty(0, 2, input_fea.shape[1], input_fea.shape[2]).cuda()
    total_feature = torch.empty(0, 1024, 40)
    for i in range(0, input_fea.shape[0] - 1):
        for j in range(i + 1, input_fea.shape[0]):
            example_fea = input_fea[i]
            target_fea = input_fea[j]
            temp = torch.stack((input_fea[i], input_fea[j]), 0) #(496,2,xx,xx)
            temp = torch.unsqueeze(temp, 0)
            group_feature = torch.cat((group_feature, temp), 0)
    return group_feature

### ARC-II MODEL

class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.encoder_backbone = encoder_edtcn()
        self.mlp = my_fc()

        self.conv1d = nn.Conv1d(512, 256, kernel_size=1)
        self.conv2d_1 = nn.Conv2d(512, 32, kernel_size=(1, 1))
        self.conv2d_2 = nn.Conv2d(32, 8, kernel_size=(1, 1))
        self.pool2d_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2d_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc_1 = nn.Linear(64, 32)
        self.fc_2 = nn.Linear(32, 8)
        self.fc_3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, input):
        encoder_feature = self.encoder_backbone(input)           # 32*512*40
        score_output = self.mlp(encoder_feature)                 # 32*1

        # ####pairwise model for group difference score ##############
        group_score = torch.empty(0).cuda()
        for i in range(0, encoder_feature.shape[0] - 1):
            for j in range(i + 1, encoder_feature.shape[0]):
                example_fea = encoder_feature[i]    # 512*40
                target_fea = encoder_feature[j]     # 512*40
                example_fea_t = torch.unsqueeze(example_fea, 0)    # 1*512*40
                target_fea_t = torch.unsqueeze(target_fea, 0)      # 1*512*40

                example_fea_t = self.conv1d(example_fea_t)          # 1*256*40
                target_fea_t = self.conv1d(target_fea_t)            # 1*256*40
                connected_matrix = torch.einsum('abd,acd->abc', example_fea_t, target_fea_t)   # 1*256*256
                connected_matrix = torch.reshape(connected_matrix, (1, 512, 32, -1))    # # 1*512*32*4
                connected_matrix = self.conv2d_1(connected_matrix)          # 1*32*32*4
                connected_matrix = self.pool2d_1(connected_matrix)          # 1*32*16*2
                connected_matrix = self.conv2d_2(connected_matrix)          # 1*8*16*2
                connected_matrix = self.pool2d_2(connected_matrix)          # 1*8*8*1

                connected_matrix = connected_matrix.view(64)
                connected_matrix = self.relu(self.fc_1(connected_matrix))
                connected_matrix = self.relu(self.fc_2(connected_matrix))
                connected_matrix = self.sig(self.fc_3(connected_matrix))

                group_score = torch.cat((group_score, connected_matrix), 0)

        return score_output, group_score

#
# class backbone(nn.Module):
#     def __init__(self):
#         super(backbone, self).__init__()
#         self.encoder_backbone = encoder_edtcn()
#         self.mlp = my_fc()
#
#         self.conv1d = nn.Conv1d(512, 256, kernel_size=1)
#         self.conv2d_1 = nn.Conv2d(64, 32, kernel_size=(5, 5))
#         self.conv2d_2 = nn.Conv2d(32, 8, kernel_size=(5, 5))
#         self.pool2d_1 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.pool2d_2 = nn.MaxPool2d(kernel_size=(2, 2))
#         self.fc_1 = nn.Linear(16, 8)
#         self.fc_2 = nn.Linear(8, 1)
#         # self.fc_3 = nn.Linear(8, 1)
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
#
#     def forward(self, input):
#         encoder_feature = self.encoder_backbone(input)           # 32*512*40
#         score_output = self.mlp(encoder_feature)                 # 32*1
#
#         # ####pairwise model for group difference score ##############
#         group_score = torch.empty(0).cuda()
#         for i in range(0, encoder_feature.shape[0] - 1):
#             for j in range(i + 1, encoder_feature.shape[0]):
#                 example_fea = encoder_feature[i]    # 512*40
#                 target_fea = encoder_feature[j]     # 512*40
#                 example_fea_t = torch.unsqueeze(example_fea, 0)    # 1*512*40
#                 target_fea_t = torch.unsqueeze(target_fea, 0)      # 1*512*40
#
#                 # example_fea_t = self.conv1d(example_fea_t)          # 1*256*40
#                 # target_fea_t = self.conv1d(target_fea_t)            # 1*256*40
#                 connected_matrix = torch.einsum('abc,abc->abc', example_fea_t, target_fea_t)   # 1*512*40
#                 connected_matrix = torch.reshape(connected_matrix, (1, 64, 20, -1))    # # 1*64*20*16
#                 connected_matrix = self.conv2d_1(connected_matrix)          # 1*32*16*12
#                 connected_matrix = self.pool2d_1(connected_matrix)          # 1*32*8*6
#                 connected_matrix = self.conv2d_2(connected_matrix)          # 1*8*2*2
#                 connected_matrix = self.pool2d_2(connected_matrix)          # 1*8*2*1
#
#                 connected_matrix = connected_matrix.view(16)
#                 connected_matrix = self.relu(self.fc_1(connected_matrix))
#                 connected_matrix = self.sig(self.fc_2(connected_matrix))
#                 # connected_matrix = self.sig(self.fc_3(connected_matrix))
#
#                 group_score = torch.cat((group_score, connected_matrix), 0)
#
#         return score_output, group_score







# class backbone(nn.Module):
#     def __init__(self):
#         super(backbone, self).__init__()
#         self.encoder_backbone = encoder_edtcn()
#         self.mlp = my_fc()
#
#         self.fc = nn.Linear(2*40*512, 1)
#
#     def forward(self, target, example):
#         target_feature = self.encoder_backbone(target)
#         example_feature = self.encoder_backbone(example)
#         target_output = self.mlp(target_feature)
#         example_output = self.mlp(example_feature)
#
#          total_feature = torch.cat((target_feature, example_feature), 1)
#          total_feature = total_feature.view(-1, 40960)
#          contrastive_output = self.fc(total_feature)
#
#          return target_output, example_output, contrastive_output
