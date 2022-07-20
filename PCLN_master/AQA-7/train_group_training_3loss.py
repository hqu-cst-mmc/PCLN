import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import ResNet_Features as resfea

import numpy as np
# import cv2
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models_for_ED_TCN.encoder_model import encoder_edtcn
from models_for_ED_TCN.my_fc import my_fc
from models_for_ED_TCN.backbone import backbone

from dataset_for_fea import divingDataset
# from visualize import make_dot
from scipy.stats import spearmanr
# import matplotlib.pyplot as plt   #
from plot_for_vis import plot_loss_spss
from losses import *
from config import *
from config import get_parser

env_path = './platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = env_path

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

def compute_group_score(options, input_score):
    group_score = torch.empty(0).cuda()
    for i in range(0, input_score.shape[0] - 1):
        for j in range(i + 1, input_score.shape[0]):
            tmp_label = abs(input_score[i] - input_score[j])
            group_score = torch.cat((group_score, tmp_label), 0)
    return group_score


def compute_loss(options, output, label):

    if options.loss == "MSELoss":
        criterion = nn.MSELoss()  ## MSEloss
        loss = criterion(output, label)
    elif options.loss == "Gaussian_loss":
        criterion = GaussianLoss()
        loss = criterion(output, label)
        loss = torch.sum(loss) / options.train_batch_size

    # loss = criterion(torch.log(probs), data['final_score'].cuda().float())
    return loss

def main(options):
    ## 获取train和test的数据以及标签 #######################################################
    dset_train = divingDataset(data_folder, train_file, label_file, spss_label_file)
    dset_test = divingDataset(data_folder, test_file, label_file, spss_label_file, test=1)

    train_loader = DataLoader(dset_train,
                              batch_size=options.train_batch_size,
                              shuffle=True,)
    test_loader = DataLoader(dset_test,
                             batch_size=options.test_batch_size,
                             shuffle=True,)

    use_cuda = (len(options.gpuid) >= 1)

    for param in model.parameters():
        param.requires_grad = True

    if use_cuda > 0:
        model.cuda()

    if options.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), options.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif options.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), options.learning_rate)
    elif options.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), options.learning_rate, alpha=0.99)

    scheduler = StepLR(optimizer, step_size=options.lr_steps[0], gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, min_lr=0.000001)   ## 每patience轮自动调整

    all_test_loss = []
    all_test_rho = []

    if not options.test:
        # main training loop############################################################################
        all_train_loss = []
        all_train_rho = []
        for epoch_i in range(0, options.epochs):
            logging.info("At {0}-th epoch.".format(epoch_i))
            train_loss = 0.0
            all_train_output = []
            all_labels = []
            all_spss_labels = []

            for it, train_data in enumerate(train_loader, 0):
                vid_tensor, labels, spss_label = train_data
                if use_cuda:
                    vid_tensor, labels, spss_label = Variable(vid_tensor).cuda(), Variable(labels).cuda(), Variable(spss_label).cuda()
                    labels = labels[:, np.newaxis]
                    spss_label = spss_label[:, np.newaxis]

                model.train()
                train_output, group_score_output = model(vid_tensor)

                ####对输出的S1，S2计算和输出的Sd之间的差距
                s1_s2_score = compute_group_score(options, train_output)
                loss_s1s2_sd = compute_loss(options, s1_s2_score, group_score_output)

                ####compute label group difference scores
                group_score_label = compute_group_score(options, labels)
                ####compute output group difference scores
                # group_score_output = compute_group_score(options, train_output)
                ####compute group scores loss
                loss_group = compute_loss(options, group_score_output, group_score_label)

                all_train_output = np.append(all_train_output, train_output.data.cpu().numpy()[:, 0])
                all_labels = np.append(all_labels, labels.data.cpu().numpy())
                all_spss_labels = np.append(all_spss_labels, spss_label.cpu().numpy())

                loss = compute_loss(options, train_output, labels)
                loss_total = loss_group + loss + loss_s1s2_sd
                train_loss += loss_total.item()
                # train_loss += loss.item()

                optimizer.zero_grad()    ## 梯度置0, 也就是把loss关于weight的导数变成0
                # loss = Variable(loss, requires_grad=True)   ###自定义Myloss需要
                # loss.backward()      ## 反向传播求梯度
                loss_total.backward()
                optimizer.step()     ## 更新权重参数

            scheduler.step()
            # scheduler.step(train_loss)    ##  RMsprop() reducelronplateau 需要监测的参数
            train_avg_loss = train_loss / (len(dset_train) / options.train_batch_size)

            min = np.amin(all_spss_labels)     ### 总分归一化后的标签
            max = np.amax(all_spss_labels)
            all_train_output = (all_train_output * (max - min)) + min

            rho, p_val = spearmanr(all_train_output, all_spss_labels)
            # np.save('results/train_result_diving370.npy', all_train_output)
            logging.info(
                "Average training loss value per instance is {0}, the corr is {1} at the end of epoch {2}".format(
                    train_avg_loss, rho, epoch_i))
            all_train_loss = np.append(all_train_loss, train_avg_loss)
            all_train_rho = np.append(all_train_rho, rho)
            if rho >= np.amax(all_train_rho):
                np.save('results/train_result_sync10m.npy', all_train_output)
                np.save('results/train_spss_sync10m.npy', all_spss_labels)
            if options.save:
                torch.save({
                    'epoch': epoch_i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, './models/checkpoint' + str(options.save) + '.tar')

            # # main Test loop#####################################################################################
            model.eval()
            test_loss = 0.0
            all_test_output = []
            all_labels = []
            all_spss_labels = []

            for it, test_data in enumerate(test_loader, 0):
                vid_tensor, labels, spss_label = test_data
                if use_cuda:
                    vid_tensor, labels, spss_label = Variable(vid_tensor).cuda(), Variable(labels).cuda(), Variable(spss_label).cuda()
                    labels = labels[:, np.newaxis]

                test_output, test_group_score_output = model(vid_tensor)

                ####对输出的S1，S2计算和输出的Sd之间的差距
                s1_s2_score = compute_group_score(options, test_output)
                loss_s1s2_sd = compute_loss(options, s1_s2_score, test_group_score_output)

                ####compute label group difference scores
                group_score_label = compute_group_score(options, labels)
                # group_score_output = compute_group_score(options, test_output)####compute output group difference scores
                loss_group = compute_loss(options, test_group_score_output, group_score_label)  ####compute group scores loss

                all_test_output = np.append(all_test_output, test_output.data.cpu().numpy()[:, 0])   # [:, 0]取所有集合的第0个数据
                all_labels = np.append(all_labels, labels.data.cpu().numpy())
                all_spss_labels = np.append(all_spss_labels, spss_label.data.cpu().numpy())

                loss = compute_loss(options, test_output, labels)
                loss_total = loss_group + loss + loss_s1s2_sd
                test_loss = loss_total.item()
                # test_loss += loss.item()
                # scheduler.step(test_loss)
            test_avg_loss = test_loss / (len(dset_test) / options.test_batch_size)

            all_test_output = (all_test_output * (max - min)) + min

            rho, p_val = spearmanr(all_test_output, all_spss_labels)
            m = len(all_labels)
            mse = sum(np.square(all_spss_labels - all_test_output)) / m

            # np.save('results/test_result_diving370.npy', all_test_output)
            logging.info("Average test loss value per instance is   {0}, the corr is {1}, the mse is {2} at the end of epoch {3}".format(
                test_avg_loss, rho, mse, epoch_i))
            all_test_loss = np.append(all_test_loss, test_avg_loss)
            all_test_rho = np.append(all_test_rho, rho)
            if rho >= np.amax(all_test_rho):
                np.save('results/test_result_sync10m.npy', all_test_output)
                np.save('results/test_spss_sync10m.npy', all_spss_labels)
        #######################################################################################################################
        # the last test for visualization
        model.eval()
        test_loss = 0.0
        all_test_output = []
        all_labels = []
        all_spss_labels = []
        for it, test_data in enumerate(test_loader, 0):
            vid_tensor, labels, spss_label = test_data
            if use_cuda:
                vid_tensor, labels, spss_label = Variable(vid_tensor).cuda(), Variable(labels).cuda(), Variable(spss_label).cuda()
                labels = labels[:, np.newaxis]

            test_output, test_group_score_output = model(vid_tensor)

            ####对输出的S1，S2计算和输出的Sd之间的差距
            s1_s2_score = compute_group_score(options, test_output)
            loss_s1s2_sd = compute_loss(options, s1_s2_score, test_group_score_output)

            ####compute label group difference scores
            group_score_label = compute_group_score(options, labels)
            # group_score_output = compute_group_score(options, test_output)  ####compute output group difference scores
            loss_group = compute_loss(options, test_group_score_output, group_score_label)  ####compute group scores loss

            all_test_output = np.append(all_test_output, test_output.data.cpu().numpy()[:, 0])
            all_labels = np.append(all_labels, labels.data.chhpu().numpy())
            all_spss_labels = np.append(all_spss_labels, spss_label.data.cpu().numpy())

            loss = compute_loss(options, test_output, labels)
            loss_total = loss_group + loss + loss_s1s2_sd
            test_loss = loss_total.item()
            # test_loss += loss.item()
        test_avg_loss = test_loss / (len(dset_test) / options.test_batch_size)

        all_test_output = (all_test_output * (max - min)) + min

        rho, p_val = spearmanr(all_test_output, all_spss_labels)

        m = len(all_labels)
        mse = sum(np.square(all_labels - all_test_output)) / m
        mde = sum(np.abs(all_labels - all_test_output)) / m
        logging.info("\n")
        logging.info("Average test loss value per instance is {0}, the corr is {1}, the mse is {2}, the mde is {3}".format(test_avg_loss, rho, mse, mde))
        maxrho_train = np.amax(all_train_rho)
        maxrho_test = np.amax(all_test_rho)
        logging.info("Max train corr value is {0}, Max test corr value  is {1} ".format(maxrho_train, maxrho_test))

        epoch_range = []
        for i in range(0, epoch_i+1):
            epoch_range.append(i)
        ## visualazition for loss and spss
        plot_loss_spss(epoch_range, all_train_loss, all_test_loss, all_train_rho, all_test_rho)


if __name__ == "__main__":
    # model = encoder_edtcn()

    model = backbone()

    ret = get_parser().parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(get_parser().parse_known_args()[1]))

    main(options)
