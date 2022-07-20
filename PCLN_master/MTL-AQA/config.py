import argparse

info_dir = './data_files'
feature_dir = 'MTL_res50Fea160'

## 获取train和test的数据以及标签 #######################################################

def get_parser():

    parser = argparse.ArgumentParser(description="Diving")
    parser.add_argument('--type',
                        type=str,
                        help='type of the model: single or multi branch',
                        choices=['single', 'multi'],
                        default='multi')
    parser.add_argument("--load", default=0, type=int,
                        help="Load saved network weights. 0 represent don't load; other number represent the model number")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Epochs through the data. (default=65)")
    parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float,
                        help="Learning rate of the optimization. (default=0.0001)")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size for training. (default=16)")
    parser.add_argument("--test_batch_size", default=16, type=int,
                        help="Batch size for training. (default=16)")
    parser.add_argument("--lr_steps", default=[90, 30], type=int, nargs="+",
                        help="steps to decay learning rate")
    parser.add_argument("--loss", default="MSELoss", choices=["MSELoss+L1Loss", "MSELoss", "L1Loss", "MyLoss+MSE",
                                                                 "MyLoss+MSE+L1Loss", "SmoothL1loss", "KLLoss",
                                                                    "Gaussian_loss", "Gaussian+Myloss", "Gaussian+MSELoss"],
                        help="different kinds of loss in training and testing loop")
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam", "RMSprop"],
                        help="Optimizer of choice for training. (default=RMSprop)")
    parser.add_argument("--gpuid", default=[0], nargs='+', type=str,
                        help="ID of gpu device to use. Empty implies cpu usage.")
    parser.add_argument("--only_last_layer", default=0, type=int,
                        help="whether choose to freezen the parameters for all the layers except the linear layer on the pre-trained model")
    parser.add_argument('--log_info', type=str,
                        help='info that will be displayed when logging', default='Exp1')
    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=False)
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=0)  # default=8

    return parser


