# python相关
import cv2
import os
import sys
import platform
import time
import itertools
import numpy as np
import warnings
import argparse

# pytorh相关
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential
from utils import save_img
from skimage.measure.simple_metrics import compare_psnr
import skimage
# 自定义类
from config import get_arguments
import Dataset
import SavePicture
import IMF
from VGG import VGGLoss
from networks import Enhance, get_scheduler, update_learning_rate
from utils import calculate_cos_distance,calculate_L1_L2_distance
#忽视警告
warnings.filterwarnings("ignore")

# 导入参数设置
parser = get_arguments()
opt = parser.parse_args()

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
#并行训练相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device0=torch.device("cuda:1")

#读取数据集
train_dataset = Dataset.DatasetTrain(opt)
test_dataset  = Dataset.DatasetTest(opt)

batch_train = 6
batch_test  = 1
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_train, shuffle=True,)
test_loader  = data.DataLoader(dataset=test_dataset,  batch_size=batch_test,  shuffle=False,)

#响应曲线
UPIMF   = IMF.upIMF(ev=2)
DOWNIMF = IMF.downIMF(ev=2)

step=1
model_path_L2H="checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
print('===> Building models')
model_restoration = Enhance()
model_restoration = torch.nn.DataParallel(model_restoration)
if os.path.exists(model_path_L2H):
    print('===> Success load model_L2H models')
    model_restoration = torch.load(model_path_L2H)
    opt.epoch_count=opt.nepochs+1
model_restoration.to(torch.device('cuda'))

# vgg_loss = VGGLoss().to(torch.device('cuda'))
# criterionL1  =  calculate_L1_L2_distance().to(torch.device('cuda'))
# calculate_cos_loss = calculate_cos_distance().to(torch.device('cuda'))
criterionL1 = nn.L1Loss().to(torch.device('cuda'))
# setup optimizer
optimizer_g = optim.Adam(model_restoration.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)


a = [];
b = [];
c = [];
if os.path.exists('a.npy'):
    a = np.load("a.npy").tolist()
if os.path.exists('b.npy'):
    b = np.load("b.npy").tolist()
if os.path.exists('c.npy'):
    c = np.load("c.npy").tolist()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    model_restoration.train()
    for idx ,(L, M, H, LW, HW) in enumerate(train_loader):
        L2M = UPIMF(L)
        H2M = DOWNIMF(H)
        LW2 = LW / (LW+HW);  HW2 = HW / (LW+HW);
        Fake_M = model_restoration(L/255.0, LW2, L2M/255.0, H/255.0, HW2, H2M/255.0 )

        # loss_MSE = criterionL1(Fake_M, M/255.0, 1.0) * opt.lamb_MSE
        # loss_gradient = calculate_cos_loss(Fake_M, M/255.0) * opt.lamb_gradient
        # loss_feature  = vgg_loss(Fake_M, M/255.0) * opt.lamb_feature

        loss = criterionL1(Fake_M, M/255.0)
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} ".format(epoch, idx, len(train_loader), loss.item()))
        optimizer_g.zero_grad()
        loss.backward()
        optimizer_g.step()

    update_learning_rate(net_g_scheduler, optimizer_g)


    if epoch % 10 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(model_restoration, net_g_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

    if epoch % 10 == 0:
        model_restoration.eval()
        with torch.no_grad():
            avg_psnr = 0.0; avg_ssim=0.0
            for idx, (L, M, H, LW, HW) in enumerate(test_loader):
                L2M = UPIMF(L)
                H2M = DOWNIMF(H)
                LW2 = LW / (LW + HW);
                HW2 = HW / (LW + HW);
                Fake_M = model_restoration( L/255.0, LW2, L2M / 255.0, H/255.0, HW2, H2M / 255.0 )

                SavePicture.save_from_tensor_test(Fake_M[0, :, :, :] * 255.0, './dataset/YL/' + str(idx) + '.png')

                img = Fake_M[0, :, :, :].clone()
                img = img.float().cpu().numpy()

                img0 = M[0, :, :, :].clone()/255.0
                img0 = img0.float().cpu().numpy()

                psnr = compare_psnr(img0, img, data_range=1.0)
                avg_psnr += psnr

                img22 = np.transpose(img, (1, 2, 0))
                img00 = np.transpose(img0, (1, 2, 0))

                ssim = skimage.measure.compare_ssim(img22, img00, data_range=1.0, multichannel=True)
                avg_ssim += ssim


            print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(test_loader)))
            a.append(epoch);
            b.append(avg_psnr / len(test_loader));
            c.append(avg_ssim / len(test_loader));

        np.save("a.npy", a)
        np.save("b.npy", b)
        np.save("c.npy", c)


