import numpy as np
from PIL import Image
from torch.nn import functional as F
import torch

loss_fnl2 = torch.nn.MSELoss(reduce=False, size_average=False)
loss_fnl1 = torch.nn.L1Loss(reduce=False,  size_average=False)


class calculate_L1_L2_distance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, param0):
        diff = torch.abs(x-y)
        param = param0 / 255.0
        mask1 = torch.where(diff >= param,  torch.abs(x-y), torch.zeros_like(x))
        mask2 = torch.where(diff <  param,  ( (x-y)*(x-y)+param*param ) / (2.0*param), torch.zeros_like(x))
        coefficient = torch.mean(torch.add(mask1, mask2))
        return coefficient

class calculate_cos_distance(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        a = a.view(a.shape[0], -1)
        b = b.view(b.shape[0], -1)
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        cose = a.mm(b.t())
        return torch.mean((1 - cose))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # img = img.resize((256, 256), Image.BICUBIC)
    return img

def load_img2(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
