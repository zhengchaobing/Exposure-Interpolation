import torch
import torchvision
from torch.utils import data
from torch.autograd import Variable
import cv2
import numpy as np
import os
import random

train_len=640
class DatasetTrain(data.Dataset):
    def __init__(self, opt):
        LowRoot=np.array([''])
        MidRoot = np.array([''])
        HighRoot = np.array([''])
        # for file in os.listdir(opt.test_root):
        for index in range(1,train_len+1):
            file = str(index)
            LowRoot  = np.append(LowRoot, opt.train_rootL + file +'.png')
            HighRoot = np.append(HighRoot,opt.train_rootH + file +'.png')
            MidRoot  = np.append(MidRoot, opt.train_rootM + file +'.png')

        self.LowRoot = LowRoot[1:]
        self.MidRoot = MidRoot[1:]
        self.HighRoot = HighRoot[1:]
        self.opt = opt

    def transformImage(self,image):
        opt = self.opt
        image = torchvision.transforms.ToTensor()(image)
        image = image*255.0
        Tensor = torch.FloatTensor if opt.device == 'cpu' else torch.cuda.FloatTensor
        image = Variable(image.type(Tensor))
        return image

    def WeightLow(self,image):
        # image = cv2.imread(self.opt.low_root)
        # image = cv2.resize(image, (int(self.opt.width), int(self.opt.height)))
        L1 = 0.0
        # L1 = 5.0
        # L2 = 55.0
        L2 = 55.0
        L3 = L2
        h = (L2 - image) / (L2 - L1)
        weight = 1 - 3 * h * h + 2 * h * h * h
        weight[image < L1] = 0.0
        weight[image >= L3] = 1.0
        weight = torch.from_numpy(weight)
        weight = weight.permute([2, 0, 1])
        Tensor = torch.FloatTensor if self.opt.device == 'cpu' else torch.cuda.FloatTensor
        weight = Variable(weight.type(Tensor))
        # weight[weight<0.000001] = 0.000001
        weight = weight + 0.000001
        # weight = self.WeightMax(weight)
        weight = weight * 255
        return weight

    def WeightHigh(self,image):
        # image = cv2.imread(self.opt.high_root)
        # image = cv2.resize(image, (int(self.opt.width), int(self.opt.height)))
        # H1 = 250.0
        H1 = 255.0
        # H2 = 200.0
        H2 = 200.0
        H3 = H2
        h = (image - H2) / (H1 - H2)
        weight = 1 - 3 * h * h + 2 * h * h * h
        weight[image > H1] = 0.0
        weight[image <= H3] = 1.0
        weight = torch.from_numpy(weight)
        weight = weight.permute([2, 0, 1])
        Tensor = torch.FloatTensor if self.opt.device == 'cpu' else torch.cuda.FloatTensor
        weight = Variable(weight.type(Tensor))
        # weight[weight<0.000001] = 0.000001
        weight = weight + 0.000001
        # weight = self.WeightMax(weight)
        weight = weight * 255.0
        return weight

    def WeightMax(self,weight):

        weight_copy = weight.clone()
        weight_copy[0, :, :] = torch.max(weight,dim=0)[0]
        weight_copy[1, :, :] = torch.max(weight,dim=0)[0]
        weight_copy[2, :, :] = torch.max(weight,dim=0)[0]
        return weight_copy


    def __getitem__(self, index):
        W = 128
        H = 128
        w_offset = random.randint(0, max(0, 600 - W - 1))
        h_offset = random.randint(0, max(0, 400 - H - 1))

        low  = cv2.imread(self.LowRoot[index]) [h_offset:h_offset + H, w_offset:w_offset + W, :]
        mid  = cv2.imread(self.MidRoot[index]) [h_offset:h_offset + H, w_offset:w_offset + W, :]
        high = cv2.imread(self.HighRoot[index])[h_offset:h_offset + H, w_offset:w_offset + W, :]

        weightlow  = self.WeightLow(low)
        weighthigh = self.WeightHigh(high)

        low  = self.transformImage(low)
        high = self.transformImage(high)
        mid  = self.transformImage(mid)

        return low, mid, high, weightlow, weighthigh

    def __len__(self):
        return int(train_len)

test_len=17
class DatasetTest(data.Dataset):
    def __init__(self, opt):
        LowRoot=np.array([''])
        MidRoot = np.array([''])
        HighRoot = np.array([''])
        # for file in os.listdir(opt.test_root):
        for index in range(1,test_len+1):
            file = str(index)
            LowRoot  = np.append(LowRoot, opt.test_rootL + file +'.png')
            HighRoot = np.append(HighRoot,opt.test_rootH + file +'.png')
            MidRoot  = np.append(MidRoot, opt.test_rootM + file +'.png')

        self.LowRoot  = LowRoot[1:]
        self.MidRoot  = MidRoot[1:]
        self.HighRoot = HighRoot[1:]
        self.opt = opt

    def transformImage(self,image):
        opt = self.opt
        image = torchvision.transforms.ToTensor()(image)
        image = image*255.0
        Tensor = torch.FloatTensor if opt.device == 'cpu' else torch.cuda.FloatTensor
        image = Variable(image.type(Tensor))
        return image

    def WeightLow(self,image):
        L1 = 0.0
        # L1 = 5.0
        # L2 = 55.0
        L2 = 55.0
        L3 = L2
        h = (L2 - image) / (L2 - L1)
        weight = 1 - 3 * h * h + 2 * h * h * h
        weight[image < L1] = 0.0
        weight[image >= L3] = 1.0
        weight = torch.from_numpy(weight)
        weight = weight.permute([2, 0, 1])
        Tensor = torch.FloatTensor if self.opt.device == 'cpu' else torch.cuda.FloatTensor
        weight = Variable(weight.type(Tensor))
        # weight[weight<0.000001] = 0.000001
        weight = weight + 0.000001
        # weight = self.WeightMax(weight)
        weight = weight * 255
        return weight

    def WeightHigh(self,image):
        H1 = 255.0
        # H2 = 200.0
        H2 = 200.0
        H3 = H2
        h = (image - H2) / (H1 - H2)
        weight = 1 - 3 * h * h + 2 * h * h * h
        weight[image > H1] = 0.0
        weight[image <= H3] = 1.0
        weight = torch.from_numpy(weight)
        weight = weight.permute([2, 0, 1])
        Tensor = torch.FloatTensor if self.opt.device == 'cpu' else torch.cuda.FloatTensor
        weight = Variable(weight.type(Tensor))
        # weight[weight<0.000001] = 0.000001
        weight = weight + 0.000001
        # weight = self.WeightMax(weight)
        weight = weight * 255.0
        return weight

    def WeightMax(self,weight):

        weight_copy = weight.clone()
        weight_copy[0, :, :] = torch.max(weight,dim=0)[0]
        weight_copy[1, :, :] = torch.max(weight,dim=0)[0]
        weight_copy[2, :, :] = torch.max(weight,dim=0)[0]
        return weight_copy


    def __getitem__(self, index):
        low  = cv2.imread(self.LowRoot[index])
        mid  = cv2.imread(self.MidRoot[index])
        high = cv2.imread(self.HighRoot[index])

        weightlow = self.WeightLow(low)
        weighthigh = self.WeightHigh(high)

        low  = self.transformImage(low)
        high = self.transformImage(high)
        mid  = self.transformImage(mid)

        return low, mid, high, weightlow, weighthigh

    def __len__(self):
        return int(test_len)