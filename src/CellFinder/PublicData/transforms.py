import numpy as np
import random
import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F


# 填充
def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


# 打包
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# 调整最小边
class RandomResize(object):
    # (n_height, n_width)
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):

        # self.min_size 是目标高度，self.max_size 是目标宽度。
        # torchvision 中，图像的尺寸通常以 (height, width) 的形式表示，与 PIL 库中的 (width, height) 有所不同
        image = F.resize(image, (self.min_size, self.max_size), interpolation=T.InterpolationMode.BICUBIC)
        # 标签也缩放
        target = F.resize(target, (self.min_size, self.max_size), interpolation=T.InterpolationMode.NEAREST)
       
        return image, target


# 随机翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


# 裁剪
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 检查图像的尺寸是否小于目标尺寸，并在必要时对图像进行填充（padding）以确保图像尺寸至少与目标尺寸一样大。
        image = pad_if_smaller(image, self.size)
        # 相同的操作
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target



# 张量
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


# 标准化
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
