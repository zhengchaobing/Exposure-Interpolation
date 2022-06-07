# python相关
import os
import warnings
# pytorh相关
import torch
from torch.utils import data
from skimage.measure.simple_metrics import compare_psnr
import numpy as np
# 自定义类
from config import get_arguments
import Dataset
import SavePicture
import IMF
import skimage
#忽视警告
warnings.filterwarnings("ignore")

# 导入参数设置
parser = get_arguments()
opt = parser.parse_args()

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
#并行训练相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device("cuda:0")

#读取数据集

test_dataset  = Dataset.DatasetTest(opt)
batch_test  = 1
test_loader  = data.DataLoader(dataset=test_dataset,  batch_size=batch_test,  shuffle=False,)

#响应曲线
UPIMF   = IMF.upIMF(ev=2)
DOWNIMF = IMF.downIMF(ev=2)

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
net_g = torch.load(model_path).to(device)

net_g.eval()
with torch.no_grad():
        avg_psnr = 0.0; avg_ssim=0.0
        for idx, (L, M, H, LW, HW) in enumerate(test_loader):
            print(idx)
            L2M = UPIMF(L)
            H2M = DOWNIMF(H)
            LW2 = LW / (LW + HW);
            HW2 = HW / (LW + HW);
            Fake_M = net_g( L/255.0, LW2, L2M / 255.0, H/255.0, HW2, H2M / 255.0 )

            SavePicture.save_from_tensor_test(Fake_M[0, :, :, :] * 255.0, './proposed17/' + str(idx+1) + '.png')

            img = Fake_M[0, :, :, :].clone()
            img = img.float().cpu().numpy()

            img0 = M[0, :, :, :].clone()/255.0
            img0 = img0.float().cpu().numpy()

            img22 = np.transpose(img, (1, 2, 0))
            img00 = np.transpose(img0, (1, 2, 0))

            psnr = compare_psnr(img0, img, data_range=1.0)
            avg_psnr += psnr

            ssim = skimage.measure.compare_ssim(img00, img22, data_range=1.0, multichannel=True)
            avg_ssim += ssim

        print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(test_loader)))
        print("===> Avg. SSIM: {:.6f} dB".format(avg_ssim / len(test_loader)))