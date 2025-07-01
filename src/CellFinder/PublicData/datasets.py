import torch
import torch.utils.data
import numpy as np
import os
import transforms as T

from PIL import Image


# 训练集
class SegmentationPresetTrain:
    def __init__(self, width, height, crop = 512, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        # 随机选择缩放比例
        scale = np.random.uniform(0.5, 2.0)

        # 新的宽与高
        n_width = int(width * scale)
        n_height = int(height * scale)

        trans = [T.RandomResize(n_height, n_width)]
        trans.extend([
            # 裁剪
            T.RandomCrop(crop),
            # 张量
            T.ToTensor(),
            # 标准化
            T.Normalize(mean=mean, std=std),
        ])

        if hflip_prob > 0:
            # 随机反转
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        # 打包
        self.transforms = T.Compose(trans)


    def __call__(self, img, target):
        return self.transforms(img, target)



# 训练集
class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, cityscapes_data_path, cityscapes_meta_path):
        self.img_dir = cityscapes_data_path + "images/"
        self.label_dir = cityscapes_meta_path + "annotations/"

        self.examples = []

        train_img_dir_path = self.img_dir
        label_img_dir_path = self.label_dir

        file_names = os.listdir(train_img_dir_path)
        for file_name in file_names:

            img_path = train_img_dir_path + file_name
            label_img_path = label_img_dir_path + file_name

            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path

            self.examples.append(example)

        self.num_examples = len(self.examples)


    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        img = Image.open(img_path).convert('RGB') # (shape: (H , W, 3))

        label_img_path = example["label_img_path"]
        label_img = Image.open(label_img_path) # (shape: (H , W))

        # 获取原始图像大小(width, height)
        original_width, original_height = img.size

        transforms_train = SegmentationPresetTrain(original_width, original_height)
        img, label_img = transforms_train(img, label_img)

        return (img, label_img)
    

    def __len__(self):
        return self.num_examples
