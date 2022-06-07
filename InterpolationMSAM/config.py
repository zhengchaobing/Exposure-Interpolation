import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser()

    # 系统环境
    parser.add_argument('--separator', help='路径分割线', default='\\')
    parser.add_argument('--device',    help='选择占用的GPU', default=torch.device("cuda:0"))

    # 测试设置
    # parser.add_argument('--test_rootH', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Test/Hig/')
    # parser.add_argument('--test_rootL', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Test/Low/')
    # parser.add_argument('--test_rootM', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Test/Mid/')
    parser.add_argument('--test_rootH', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Test_Data/Hig/')
    parser.add_argument('--test_rootL', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Test_Data/Low/')
    parser.add_argument('--test_rootM', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Test_Data/Low/')



    parser.add_argument('--train_rootH', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Train/Hig/')
    parser.add_argument('--train_rootL', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Train/Low/')
    parser.add_argument('--train_rootM', help='测试图像路径',  default='/netdisk-2.2-home/ZCB/A_Paper_TCP_Interpolation/Data/Train/Mid/')


    parser.add_argument('--test_num',   help='测试图像对数量', type=int,default=400)
    parser.add_argument('--DisplayDirTest', help='测试结果路径', type=str, default='./data/YL/')
    parser.add_argument('--DisplayType',    help='测试图像类型', type=str,default='.png')

    parser.add_argument('--dataset', type=str, default='./dataset1', help='facades')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=32, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=32, help='discriminator filters in first conv layer')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=1000000, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr',      type=float, default=0.00001, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--beta1',   type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda',    action='store_true', default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed',    type=int, default=61, help='random seed to use. Default=123')
    parser.add_argument('--lamb',    type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--lamb_MSE', type=int, default=2.0, help='weight on L1 term in objective')
    parser.add_argument('--lamb_feature', type=int, default=0.01, help='weight on L1 term in objective')
    parser.add_argument('--lamb_gradient', type=int, default=0.01, help='weight on L1 term in objective')
    parser.add_argument('--nepochs', type=int, default=850, help='restare to train saved model of which epochs')

    return parser

