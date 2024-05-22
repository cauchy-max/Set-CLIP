# -*- coding: utf-8 -*-
# @Time    : 2023/11/10
# @Author  : MinerSong
import os
import cv2
import torch
import albumentations as A
import numpy as np
from transformers import BertTokenizer, AutoImageProcessor
from PIL import Image

from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)



class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img




def ContrastiveLearningViewGenerator(x, base_transform):
    """Take two random crops of one image as the query and key."""

    return base_transform(x)

# class ContrastiveLearningViewGenerator(object):
#     """Take two random crops of one image as the query and key."""

#     def __init__(self, base_transform, n_views=2):
#         self.base_transform = base_transform
#         self.n_views = n_views

#     def __call__(self, x):
#         return base_transform(x)





def get_simclr_pipeline_transform(s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomResizedCrop(size=224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * 224)),
                                            transforms.ToTensor()
                                            ])
    return data_transforms




mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_simclr_pipeline_transform_vit(s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomResizedCrop(size=224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * 224)),
                                            transforms.ToTensor()
                                            # transforms.Normalize(mean=mean, std=std)
                                            ])
    return data_transforms





class Collator(object):
    def __init__(self,
                 tokenizer,
                 features=("input_ids", "token_type_ids", "attention_mask", "image", "caption"),
                 max_len=200):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len
        self.image_processor = AutoImageProcessor.from_pretrained("./vit_pretrained_weight")

    def collate(self, batch):
        # print(batch)
        new_batch = []
        for example in batch:
            # 每个句子重复两次
            new_batch.append({fea: example[fea] for fea in self.features})
            temp={fea: example[fea] for fea in self.features}
            temp['image']=ContrastiveLearningViewGenerator(temp['image'],get_simclr_pipeline_transform())
            new_batch.append(temp)
        
        # 我们的dataset中应该已经padding过了
        # new_batch = self.tokenizer.pad(
        #     new_batch,
        #     padding=True,
        #     max_length=self.max_len,
        #     return_tensors="pt"
        # )
        return new_batch
    
    def collate_semantic(self, batch):
        new_batch = []
        for example in batch:
            # 每个句子重复两次
            # new_batch.append({fea: example[fea] for fea in self.features})
            temp={fea: example[fea] for fea in self.features}
            temp['image']=ContrastiveLearningViewGenerator(temp['image'],get_simclr_pipeline_transform())
            new_batch.append(temp)
        
        # 我们的dataset中应该已经padding过了
        # new_batch = self.tokenizer.pad(
        #     new_batch,
        #     padding=True,
        #     max_length=self.max_len,
        #     return_tensors="pt"
        # )
        return new_batch
    
    def collate_vit(self, batch):
        # print(batch)
        new_batch = []
        for example in batch:
            # 每个句子重复两次
            # 处理图像
            # example['image'] = self.image_processor(images=example['image'], return_tensors="pt")
            new_batch.append({fea: example[fea] for fea in self.features})
            temp={fea: example[fea] for fea in self.features}
            temp['image']=ContrastiveLearningViewGenerator(temp['image'],get_simclr_pipeline_transform_vit())
            # temp['image'] = self.image_processor(images=temp['image'], return_tensors="pt")
            new_batch.append(temp)
        
        # 我们的dataset中应该已经padding过了
        # new_batch = self.tokenizer.pad(
        #     new_batch,
        #     padding=True,
        #     max_length=self.max_len,
        #     return_tensors="pt"
        # )
        return new_batch


if __name__ == '__main__':
    pass