import os
import cv2
import torch
import albumentations as A
import numpy as np
from transformers import BertTokenizer, AutoImageProcessor

from torch import nn
from torchvision.transforms import transforms

from Collator import *
from CLIP_plus import *

np.random.seed(0)

# labels = torch.cat([torch.arange(32) for i in range(2)], dim=0)
# labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
# print(labels[32])

device = torch.device('cuda:7' if torch.cuda.is_available else 'cpu')


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        # 传进来的image_filenames和captions应该是array的形式，要注意啊！！！！！
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # image = cv2.imread(f"./mscoco/Images/{self.image_filenames[idx]}")
        image = cv2.imread(f"./archive/images/{self.image_filenames[idx]}")
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # 这个caption貌似没有实际作用，其实就是为了方便检查罢了
        # item里应该有image/caption/input_ids/attention_mask/token_type_ids，只不过caption没用吧
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



# 加载预训练的图像处理器
image_processor = AutoImageProcessor.from_pretrained("./vit_pretrained_weight")

class CLIPDataset_vit(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        # 传进来的image_filenames和captions应该是array的形式，要注意啊！！！！！
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=200
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # image = cv2.imread(f"./mscoco/Images/{self.image_filenames[idx]}")
        image = cv2.imread(f"./archive/images/{self.image_filenames[idx]}")
        
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        # image = image_processor(images=image, return_tensors="pt")
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # 这个caption貌似没有实际作用，其实就是为了方便检查罢了
        # item里应该有image/caption/input_ids/attention_mask/token_type_ids，只不过caption没用吧
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)






if __name__ == '__main__':
    with open("./archive/captions.txt", "r") as f:
        data = f.readlines()
        image_list=[]
        for i in range(1,len(data)):
            temp=data[i].split(".jpg")
            image_list.append(temp[0]+".jpg")

        image_array=np.array(image_list)
        # print(image_array)


        caption_list=[]
        for i in range(1,len(data)):
            temp=data[i].split(".jpg,")
            caption_list.append(temp[1].replace("\n",""))

        caption_array=np.array(caption_list)
        # print(caption_array)


        # 测试dataset
        transforms = get_transforms(mode="train")
        tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")

        dataset = CLIPDataset(
            image_array,
            caption_array,
            tokenizer=tokenizer,
            transforms=transforms,
        )
        # 再这里dataset还是在cpu上，dataloader可以到gpu上，且在CLIP_plus中，我已经将数据又放到gpu上了

        print(dataset[50])

        CLIP = CLIP_plus().to(device)
        a,b = CLIP(dataset)
        print(b.shape)


        # collator = Collator(1)
        # t=collator.collate(dataset)
        # print(t)


    pass