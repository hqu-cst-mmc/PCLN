import argparse

def get_parser():

    parser = argparse.ArgumentParser(description="Diving")
    parser.add_argument('--type', type=str, help='type of the model sport',
                        choices=['diving370', 'gymvault', 'skiing', 'snowboard', 'sync3m', 'sync10m'],
                        default='diving370')
    parser.add_argument("--save", default=0, type=int,
                        help="Save network weights. 0 represent don't save; number represent model number")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Epochs through the data. (default=65)")
    parser.add_argument("--learning_rate", "-lr", default=0.0001, type=float,
                        help="Learning rate of the optimization. (default=0.0001)")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training. (default=16)")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training. (default=16)")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for training. (default=16)")
    parser.add_argument("--lr_steps", default=[90, 30], type=int, nargs="+",
                        help="steps to decay learning rate")
    parser.add_argument("--loss", default="MSELoss", choices=["MSELoss+L1Loss", "MSELoss", "L1Loss", "MyLoss+MSE",
                                                                 "MyLoss+MSE+L1Loss", "SmoothL1loss", "KLLoss", "Gaussian_loss"],
                        help="different kinds of loss in training and testing loop")
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam", "RMSprop"],
                        help="Optimizer of choice for training. (default=RMSprop)")
    parser.add_argument("--gpuid", default=[0], nargs='+', type=str,
                        help="ID of gpu device to use. Empty implies cpu usage.")
    parser.add_argument("--size", default=160, type=int,
                        help="size of images.")
    parser.add_argument("--test", default=0, type=int,
                        help="whether get into the whole test mode (not recommend) ")

    return parser

# Path to the directories of features and labels

# #####diving370#############
data_folder = './diving370_resFea160'
train_file = './data_files/training_idx_diving370.npy'
test_file = './data_files/testing_idx_diving370.npy'
label_file = './data_files/overscore_norm_diving370.npy'
spss_label_file = './data_files/overall_scores_diving370.npy'

# # # #######gymvault#############
# data_folder = './gymvault_resFea160'
# train_file = './data_files/training_idx_gymvault.npy'
# test_file = './data_files/testing_idx_gymvault.npy'
# label_file = './data_files/overscore_norm_gymvault.npy'
# spss_label_file = './data_files/overall_scores_gymvault.npy'

# #####skiing#############
# data_folder = './skiing_resFea160'
# train_file = './data_files/training_idx_skiing.npy'
# test_file = './data_files/testing_idx_skiing.npy'
# label_file = './data_files/overscore_norm_skiing.npy'
# spss_label_file = './data_files/overall_scores_skiing.npy'

# # ######snowboard#############
# data_folder = './snowboard_resFea160'
# train_file = './data_files/training_idx_snowboard.npy'
# test_file = './data_files/testing_idx_snowboard.npy'
# label_file = './data_files/overscore_norm_snowboard.npy'
# spss_label_file = './data_files/overall_scores_snowboard.npy'

#####sync3m#############
# data_folder = './sync3m_resFea160'
# train_file = './data_files/training_idx_sync3m.npy'
# test_file = './data_files/testing_idx_sync3m.npy'
# label_file = './data_files/overscore_norm_sync3m.npy'
# spss_label_file = './data_files/overall_scores_sync3m.npy'

######sync10m#############
# data_folder = './sync10m_resFea160'
# train_file = './data_files/training_idx_sync10m.npy'
# test_file = './data_files/testing_idx_sync10m.npy'
# label_file = './data_files/overscore_norm_sync10m.npy'
# spss_label_file = './data_files/overall_scores_sync10m.npy'