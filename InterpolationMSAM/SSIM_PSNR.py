import numpy as np
from PIL import Image
import os
import argparse
from glob import glob
import random

import test # 先进行图片的生成

def psnr_scaled(im1, im2): # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 **2 / mse)
    return psnr

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

psnr_sum = 0
# test_low_data_name = glob(os.path.join('/netdisk_home/zcb/PycharmProjects/Paper_1/Denoise_Net/dataset2/test/False') + '/*.*')
test_low_data_name = glob(os.path.join('E:\\Dataset\\New_data\\New_test\\YL\\') + '*.*')
test_low_data = []
for idx in range(len(test_low_data_name)):
    test_low_im = load_images(test_low_data_name[idx])  # 读取数据并转化为【0 1】之间
    test_low_data.append(test_low_im)  # 把图像数据放进列表

for idx in range(len(test_low_data)):
    [_, name] = os.path.split(test_low_data_name[idx])
    name = name[:name.find('.')]
    # clean_image = np.array(Image.open(os.path.join('/netdisk_home/zcb/PycharmProjects/Paper_1/Denoise_Net/dataset2/test/Truth/', name + "." + "jpg")), dtype="float32") / 255.0
    clean_image = np.array(Image.open(os.path.join('E:\\Dataset\\New_data\\New_test\\Mid\\', name + "." + "png")), dtype="float32") / 255.0
    psnr = psnr_scaled(clean_image, test_low_data[idx])
    psnr_sum += psnr

avg_psnr = psnr_sum / len(test_low_data)
print('YL psnr= {}'.format(avg_psnr))

psnr_sum = 0
# test_low_data_name = glob(os.path.join('/netdisk_home/zcb/PycharmProjects/Paper_1/Denoise_Net/dataset2/test/False') + '/*.*')
test_low_data_name = glob(os.path.join('E:\\Dataset\\New_data\\New_test\\YY\\') + '*.*')
test_low_data = []
for idx in range(len(test_low_data_name)):
    test_low_im = load_images(test_low_data_name[idx])  # 读取数据并转化为【0 1】之间
    test_low_data.append(test_low_im)  # 把图像数据放进列表

for idx in range(len(test_low_data)):
    [_, name] = os.path.split(test_low_data_name[idx])
    name = name[:name.find('.')]
    # clean_image = np.array(Image.open(os.path.join('/netdisk_home/zcb/PycharmProjects/Paper_1/Denoise_Net/dataset2/test/Truth/', name + "." + "jpg")), dtype="float32") / 255.0
    clean_image = np.array(Image.open(
        os.path.join('E:\\Dataset\\New_data\\New_test\\Mid\\', name + "a." + "png")),
                           dtype="float32") / 255.0
    psnr = psnr_scaled(clean_image, test_low_data[idx])
    psnr_sum += psnr

avg_psnr = psnr_sum / len(test_low_data)
print('YY psnr= {}'.format(avg_psnr))