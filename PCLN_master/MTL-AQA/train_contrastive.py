import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import itertools
import numpy as np
# import cv2
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from models_for_ED_TCN.encoder_model import encoder_edtcn
from models_for_ED_TCN.encoder_model_contrastive import encoder_edtcn
from models_for_ED_TCN.my_fc import my_fc
from models_for_ED_TCN.backbone import backbone

from dataset_for_contrastive import VideoDataset
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

env_path = './platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = env_path

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def get_judge_score_label(options, data):
    judge_score = Variable(data['judge_scores']).cuda()
    judge_scores_sort = torch.sort(judge_score)[0] / 10
    ## only use 3 judge scores as loss label
    judge_scores_sort = judge_scores_sort[:, 2:5]

    # judge_label = torch.sum(judge_label, dim=1)
    # judge_label = torch.sum(judge_scores_sort[:, 2:5], dim=1)     ##sum of 3 judge scores

    return judge_scores_sort

def compute_score(options, output):
    ### calculate expectation & denormalize
    max = np.amax(output)
    min = np.amin(output)
    all_output = (output * (max - min)) + min
    return all_output

def compute_loss(options, output, label):
    if options.loss == "MSELoss":
        criterion = nn.MSELoss()  ## MSEloss
        loss = criterion(output, label)
    elif options.loss == "Gaussian_loss":
        criterion = GaussianLoss()
        # ## compute loss with execution score
        loss = criterion(output, label)
        # loss = torch.sum(loss) / 3 / options.train_batch_size
        loss = torch.sum(loss) / options.train_batch_size
    return loss

def get_dataloaders(options):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', options),
                                                       batch_size=options.train_batch_size,
                                                       num_workers=options.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', options),
                                                      batch_size=options.test_batch_size,
                                                      num_workers=options.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, options):
    use_cuda = (len(options.gpuid) >= 1)
    # parameters_2_optimize = model_contrastive.parameters()
    # parameters_2_optimize = (list(model_encoder.parameters()) + list(model_fc.parameters()) +
    #                          list(model_contrastive.parameters()))
    for param in model_contrastive.parameters():
        param.requires_grad = True

    if use_cuda > 0:
        # model_encoder.cuda()
        # model_fc.cuda()
        model_contrastive.cuda()

    if options.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_contrastive.parameters(), options.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif options.optimizer == "Adam":
        optimizer = torch.optim.Adam(model_contrastive.parameters(), options.learning_rate)
    elif options.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model_contrastive.parameters(), options.learning_rate, alpha=0.99)

    scheduler = StepLR(optimizer, step_size=options.lr_steps[0], gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.35, min_lr=0.000001, patience=32)

    all_train_loss, all_train_rho, all_test_loss, all_test_rho = [], [], [], []

    for epoch_i in range(options.epochs):
        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss = 0.0
        all_train_output1, all_train_output2 = [], []
        all_diffs1, all_diffs2 = [], []
        all_labels1, all_labels2 = [], []
        all_spss_labels1, all_spss_labels2 = [], []
        all_norm_score1, all_norm_score2 = [], []
        all_contra_label = []
        #
        # main training loop##############################################################################################
        for it, (data, target) in enumerate(dataloaders['train']):
            fea_tensor_1 = Variable(data['feature']).cuda()
            true_scores_1 = Variable(data['final_score']).cuda()
            norm_score_1 = Variable(data['norm_score']).cuda()
            diff_degree_1 = Variable(data['difficulty']).cuda()
            judge_label_1 = get_judge_score_label(options, data)
            norm_judge_score_1 = Variable(data['norm30_judge_scores']).cuda()

            # judge_label = judge_label[:, np.newaxis]
            norm_judge_score_1 = norm_judge_score_1[:, np.newaxis].float()
            true_scores_1 = true_scores_1[:, np.newaxis].float()
            norm_score_1 = norm_score_1[:, np.newaxis].float()
            diff_degree_1 = diff_degree_1[:, np.newaxis].float()

            fea_tensor_2 = Variable(target['feature']).cuda()
            true_scores_2 = Variable(target['final_score']).cuda()
            norm_score_2 = Variable(target['norm_score']).cuda()
            diff_degree_2 = Variable(target['difficulty']).cuda()
            judge_label_2 = get_judge_score_label(options, target)
            norm_judge_score_2 = Variable(target['norm30_judge_scores']).cuda()

            norm_judge_score_2 = norm_judge_score_2[:, np.newaxis].float()
            true_scores_2 = true_scores_2[:, np.newaxis].float()
            norm_score_2 = norm_score_2[:, np.newaxis].float()
            diff_degree_2 = diff_degree_2[:, np.newaxis].float()

            contra_label = norm_judge_score_1 - norm_judge_score_2

            # model_encoder.train()
            # model_fc.train()
            model_contrastive.train()

            # encoder_feature_1 = model_encoder(fea_tensor)
            # train_output_1 = model_fc(encoder_feature_1)      ## in: features   out: scores
            score_output_2, score_output_1, contra_output = model_contrastive(fea_tensor_2, fea_tensor_1)

            all_train_output1 = np.append(all_train_output1, score_output_1.data.cpu().numpy()[:, 0])
            all_train_output2 = np.append(all_train_output2, score_output_2.data.cpu().numpy()[:, 0])
            all_diffs1 = np.append(all_diffs1, diff_degree_1.data.cpu().numpy())
            all_diffs2 = np.append(all_diffs2, diff_degree_2.data.cpu().numpy())
            all_spss_labels1 = np.append(all_spss_labels1, true_scores_1.data.cpu().numpy())
            all_spss_labels2 = np.append(all_spss_labels2, true_scores_2.data.cpu().numpy())
            all_norm_score1 = np.append(all_norm_score1, norm_judge_score_1.data.cpu().numpy())
            all_norm_score2 = np.append(all_norm_score2, norm_judge_score_2.data.cpu().numpy())
            # all_contra_label = np.abs(all_norm_score2 - all_norm_score1)
            # all_norm_score = np.append(all_norm_score, norm_score.data.cpu().numpy())

            loss1 = compute_loss(options, score_output_1, norm_judge_score_1)
            loss2 = compute_loss(options, score_output_2, norm_judge_score_2)   ##use judge scores as label and compute loss
            loss_contra = compute_loss(options, contra_output, contra_label)
            # loss = compute_loss(options, train_output, norm_score)    ##use overall score as label and compute loss
            loss = loss1 + loss2 + loss_contra
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # loss.backward(torch.ones_like(train_output))
            optimizer.step()
        scheduler.step()

        train_avg_loss = train_loss / (len(VideoDataset('train', options))) / options.train_batch_size

        # min = np.amin(all_spss_labels)  ### 总分归一化后的标签
        # max = np.amax(all_spss_labels)
        # all_train_output = (all_train_output * (max - min)) + min
        all_train_output1 = 30 * all_train_output1 * all_diffs1
        all_train_output2 = 30 * all_train_output2 * all_diffs2
        # rho, p = spearmanr(all_train_output, all_spss_labels)

        # np.save('./results/es_train_result.npy', all_train_output)
        # np.save('./results/es_train_src_', all_spss_labels)

        logging.info(
            "Average training loss value per instance is {0},the corr is {1} at the end of epoch {2}".format(
                train_avg_loss, rho, epoch_i))
        all_train_loss = np.append(all_train_loss, train_avg_loss)
        all_train_rho = np.append(all_train_rho, rho)

        # # main Test loop####################################################################################################################
        # model_encoder.eval()
        # model_fc.eval()
        model_contrastive.eval()

        test_loss = 0.0
        all_test_output = []
        all_test_spss_labels = []
        all_test_diffs = []
        all_test_norm_score = []

        for it, (data, target) in enumerate(dataloaders['test']):
            fea_tensor = Variable(data['feature']).cuda()
            true_scores = Variable(data['final_score']).cuda()
            norm_score = Variable(data['norm_score']).cuda()
            diff_degree = Variable(data['difficulty']).cuda()
            judge_label = get_judge_score_label(options, data)
            norm_judge_score = Variable(data['norm_judge_scores']).cuda()

            # judge_label = judge_label[:, np.newaxis].float()
            norm_judge_score = norm_judge_score[:, np.newaxis].float()
            true_scores = true_scores[:, np.newaxis].float()
            norm_score = norm_score[:, np.newaxis].float()

            fea_tensor_2 = Variable(target['feature']).cuda()
            true_scores_2 = Variable(target['final_score']).cuda()
            norm_score_2 = Variable(target['norm_score']).cuda()
            diff_degree_2 = Variable(target['difficulty']).cuda()
            judge_label_2 = get_judge_score_label(options, target)
            norm_judge_score_2 = Variable(target['norm30_judge_scores']).cuda()

            # judge_label_2 = judge_label[:, np.newaxis]
            norm_judge_score_2 = norm_judge_score[:, np.newaxis].float()
            true_scores_2 = true_scores[:, np.newaxis].float()
            norm_score_2 = norm_score[:, np.newaxis].float()
            diff_degree_2 = diff_degree[:, np.newaxis].float()

            test_encoder_feature = model_encoder(fea_tensor)
            test_output = model_fc(test_encoder_feature)

            all_test_output = np.append(all_test_output, test_output.data.cpu().numpy()[:, 0])
            all_test_spss_labels = np.append(all_test_spss_labels, true_scores.data.cpu().numpy())
            all_test_diffs = np.append(all_test_diffs, diff_degree.data.cpu().numpy())
            all_test_norm_score = np.append(all_test_norm_score, norm_judge_score.data.cpu().numpy())
            # all_norm_score = np.append(all_norm_score, norm_score.data.cpu().numpy())

            # test_output = 30 * test_output       ##use judge scores as label, but use final score to compute loss
            # loss = compute_loss(options, test_output, judge_label)
            loss = compute_loss(options, test_output, norm_judge_score)   ##use judge scores as label and compute loss
            # loss = compute_loss(options, test_output, norm_score)
            test_loss += loss.item()

        test_avg_loss = test_loss / (len(VideoDataset('test', options))) / options.test_batch_size

        # all_test_output = (all_test_output * (max - min)) + min
        all_test_output = 30 * all_test_output * all_test_diffs       ##执行分数训练的逆归一化
        rho, p_val = spearmanr(all_test_output, all_test_spss_labels)
        logging.info(
            "Average test loss value per instance is     {0}, the corr is {1} at the end of epoch {2}".format(
            test_avg_loss, rho, epoch_i))
        all_test_loss = np.append(all_test_loss, test_avg_loss)
        all_test_rho = np.append(all_test_rho, rho)
        if rho >= np.amax(all_test_rho):
            np.save('./result_plot/os_test_diff.npy', all_test_diffs)
            np.save('./result_plot/os_test_result.npy', all_test_output)
            np.save('./result_plot/os_test_src.npy', all_test_spss_labels)

    #############################################################################################################################
    ## the last test for visualization
    # model_encoder.eval()
    # model_fc.eval()
    model_contrastive.eval()

    test_loss = 0.0
    all_test_output = []
    all_labels = []
    all_diffs = []

    for it, data in enumerate(dataloaders['test']):
        fea_tensor = Variable(data['feature']).cuda()
        true_scores = Variable(data['final_score']).cuda()
        norm_score = Variable(data['norm_score']).cuda()
        diff_degree = Variable(data['difficulty']).cuda()
        judge_label = get_judge_score_label(options, data)
        norm_judge_score = Variable(data['norm_judge_scores']).cuda()

        norm_judge_score = norm_judge_score[:, np.newaxis].float()
        # judge_label = judge_label[:, np.newaxis].float()
        true_scores = true_scores[:, np.newaxis].float()
        norm_score = norm_score[:, np.newaxis].float()
        diff_degree = diff_degree[:, np.newaxis].float()

        fea_tensor_2 = Variable(target['feature']).cuda()
        true_scores_2 = Variable(target['final_score']).cuda()
        norm_score_2 = Variable(target['norm_score']).cuda()
        diff_degree_2 = Variable(target['difficulty']).cuda()
        judge_label_2 = get_judge_score_label(options, target)
        norm_judge_score_2 = Variable(target['norm30_judge_scores']).cuda()

        # judge_label_2 = judge_label[:, np.newaxis]
        norm_judge_score_2 = norm_judge_score[:, np.newaxis].float()
        true_scores_2 = true_scores[:, np.newaxis].float()
        norm_score_2 = norm_score[:, np.newaxis].float()
        diff_degree_2 = diff_degree[:, np.newaxis].float()

        test_encoder_feature = model_encoder(fea_tensor)
        test_output = model_fc(test_encoder_feature)

        all_test_output = np.append(all_test_output, test_output.data.cpu().numpy()[:, 0])
        all_labels = np.append(all_labels, true_scores.data.cpu().numpy())
        all_diffs = np.append(all_diffs, diff_degree.data.cpu().numpy())
        all_norm_score = np.append(all_norm_score, norm_judge_score.data.cpu().numpy())

        # test_output = 30 * test_output   ##use judge scores as label, but use final score to compute loss
        # loss = compute_loss(options, test_output, judge_label)
        loss = compute_loss(options, test_output, norm_judge_score)    ##use judge scores as label and compute loss
        # loss = compute_loss(options, test_output, norm_score)
        test_loss += loss.item()

    test_avg_loss = test_loss / (len(VideoDataset('test', options))) / options.test_batch_size

    # all_test_output = (all_test_output * (max - min)) + min
    all_test_output = 30 * all_test_output * all_diffs    ##难度系数标签的逆归一化
    rho, p_val = spearmanr(all_test_output, all_labels)

    m = len(all_labels)
    mse = sum(np.square(all_labels - all_test_output)) / m
    mde = sum(np.abs(all_labels - all_test_output)) / m
    logging.info("\n")
    logging.info("Average test loss value per instance is {0}, the corr is {1}, the mse is {2}, the mde is {3}".format(
        test_avg_loss, rho, mse, mde))
    maxrho_train = np.amax(all_train_rho)
    maxrho_test = np.amax(all_test_rho)
    logging.info("Max train corr value is {0}, Max test corr value  is {1} ".format(maxrho_train, maxrho_test))

    epoch_range = []
    for i in range(0, epoch_i + 1):
        epoch_range.append(i)
    ## visualazition for loss and spss
    plot_loss_spss(epoch_range, all_train_loss, all_test_loss, all_train_rho, all_test_rho)


if __name__ == "__main__":
    model_encoder = encoder_edtcn()
    model_fc = my_fc()
    model_contrastive = backbone()
    # model = encoder_edtcn()

    ret = get_parser().parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(get_parser().parse_known_args()[1]))

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    base_logger = get_logger(f'exp/MTL.log', options.log_info)
    dataloaders = get_dataloaders(options)

    main(dataloaders, options)
