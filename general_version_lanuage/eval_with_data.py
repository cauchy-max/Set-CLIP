import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

from dataset import *
from CLIP_plus_copy import *
from Collator import *
from mkmmd_loss import *


device = torch.device('cuda:7' if torch.cuda.is_available else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("train_file", type=str, help="train text file")
    # parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    # parser.add_argument("--model_out", type=str, default="./model", help="model output path")
    # parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    # parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # parser.add_argument("--tao", type=float, default=0.05, help="temperature")
    # parser.add_argument("--device", type=str, default="cuda", help="device")
    # parser.add_argument("--display_interval", type=int, default=50, help="display interval")
    # parser.add_argument("--save_interval", type=int, default=100, help="save interval")
    # parser.add_argument("--pool_type", type=str, default="cls", help="pool_type")
    # parser.add_argument("--dropout_rate", type=float, default=0.3, help="dropout_rate")
    args = parser.parse_args()
    return args



def build_loaders(args, image_array, caption_array, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        image_array,
        caption_array,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    # print(dataset[1])
    collator = Collator(tokenizer)
    # 每一个dataloader里面有2*batch_size个数据
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last = True,
        collate_fn=collator.collate_semantic#这个一定要有，否则会报错，感觉是dataloader的bug
    )
    return dataloader


def eval(args,image_array,caption_array,mode="eval"):
    # tokenizer在这里定义
    tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")
    # mode在build_loaders里面才用
    dl=build_loaders(args,image_array,caption_array,tokenizer,mode)
    
    model = torch.load('model_clip_50_6000.pkl')
    model.eval()
    model.to(device)
    acc=0
    acc1=0
    # print(";;;;;;;;;;;;;;;;")


    with torch.no_grad():
        for batch in tqdm(dl):
            # print(batch)
            image_embeddings,text_embeddings = model(batch)
            # 因为做了normalize，所以这就相当于cosine similarity了
            logits = (text_embeddings @ image_embeddings.T) 
            logits1 = (image_embeddings @ text_embeddings.T) 
            targets = torch.arange(0, image_embeddings.shape[0], device=device)
            # targets = F.softmax(
            #     (images_similarity + texts_similarity) / 2 * temperature, dim=-1
            # )
            
            
            # a = torch.max(logits, dim=1, keepdim=True)[1].detach().cpu().numpy().flatten()
            # b = torch.max(logits1, dim=1, keepdim=True)[1].detach().cpu().numpy().flatten()
            values, indices = logits.topk(1, dim=1, largest=True, sorted=True)
            a = indices.detach().cpu().numpy()
            values, indices = logits1.topk(1, dim=1, largest=True, sorted=True)
            b = indices.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy().flatten()
            for i in range(50):
                if targets[i] in a[i]:
                    acc = acc+1
                if targets[i] in b[i]:
                    acc1 = acc1+1
            
            print(acc/2000)
            print(acc1/2000)
            print(acc+acc1)
    
    print(acc+acc1)
            
    
    


if __name__ == '__main__':
    args = parse_args()

    # import json
    # with open('./mscoco/captions.json') as f:
    #     superHeroSquad = json.load(f)
    #     print(len(superHeroSquad))
    #     # print(superHeroSquad[0])
    #     # print(superHeroSquad[1])
    #     image_list=[]
    #     for i in range(15000,18000):
    #         temp=str(superHeroSquad[i]['image_id'])
    #         rest_len=12-len(temp)
    #         before='0'*rest_len
    #         before='COCO_train2014_'+before
    #         temp=before+temp+'.jpg'
    #         image_list.append(temp)
        
    #     caption_list=[]
    #     for i in range(15000,18000):#len(data)):
    #         temp=superHeroSquad[i]['caption']
    #         caption_list.append(temp.replace("\n",""))
    
    
    
    
    
    # 数据处理过程
    with open("./archive/captions.txt", "r") as f:
        data = f.readlines()
        image_list=[]
        for i in range(6000,8000):
            temp=data[i].split(".jpg")
            image_list.append(temp[0]+".jpg")

        caption_list=[]
        for i in range(6000,8000):
            temp=data[i].split(".jpg,")
            caption_list.append(temp[1].replace("\n",""))




        # image_list1=[]
        # caption_list1=[]
        # for i in range(len(image_list)):
        #     image = cv2.imread(f"./archive/images/{image_list[i]}")
        #     if image is not None:
        #         image_list1.append(image_list[i])
        #         caption_list1.append(caption_list[i])
                

        print(len(image_list))
        image_array=np.array(image_list)
        # print(image_array)
        caption_array=np.array(caption_list)
        # print(caption_array)


        eval(args,image_array,caption_array,mode="eval")