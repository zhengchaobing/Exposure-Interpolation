import cv2
import torch
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def save_from_tensor_test(tensor, filename):
    img = tensor.clone()
    img = torch.clamp(img, 0, 255)
    # 反向标准化 到[0,1]区间
    img = img / 255
    img = tensor_to_np(img)
    # img = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(filename, img)

