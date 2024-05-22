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
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
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
        collate_fn=collator.collate_semantic#这个一定要有，否则会报错，感觉是dataloader的bug
    )
    return dataloader


def eval(args,image_array,caption_array,mode="eval"):
    # tokenizer在这里定义
    tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")
    # mode在build_loaders里面才用
    dl=build_loaders(args,image_array,caption_array,tokenizer,mode)
    
    model = torch.load('model_end2end_49_3000_1000.pkl')
    model.eval()
    model.to(device)
    figure_list1=[]
    flag1=1


    with torch.no_grad():
        for batch in tqdm(dl):
            # print(batch)
            image_embeddings,text_embeddings = model(batch)
            if flag1==1:
                # figure_list1=[]
                # print(image_pred.shape)
                figure_list1.append(image_embeddings)
                figure_list1.append(text_embeddings)
                flag1=1
    
    # 画图了
    a=figure_list1[0]
    # a=torch.cat((figure_list1[0], figure_list1[2]), 0)
    # print(a)
    b=figure_list1[1]
    # b=torch.cat((figure_list1[1], figure_list1[3]), 0)
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    # print(a.shape)
    reducer = umap.UMAP(n_neighbors=7, n_components=2)
    whole=np.concatenate((a,b),axis=0)
    reducer=reducer.fit(whole)
    # 使用UMAP进行降维
    embedding_a = reducer.transform(a)
    embedding_b = reducer.transform(b)

    fig1 = plt.figure(figsize=(10, 8))
    ax_compare=embedding_a[0:5,0]
    ay_compare=embedding_a[0:5,1]
    plt.plot(ax_compare,ay_compare, 'o',color='r',markersize=10)
    ax_compare=embedding_a[5:8,0]
    ay_compare=embedding_a[5:8,1]
    plt.plot(ax_compare,ay_compare, 'o',color='k',markersize=10)
    ax_compare=embedding_a[8:13,0]
    ay_compare=embedding_a[8:13,1]
    plt.plot(ax_compare,ay_compare, 'o',color='g',markersize=10)
    ax_compare=embedding_a[13:23,0]
    ay_compare=embedding_a[13:23,1]
    plt.plot(ax_compare,ay_compare, 'o',color='m',markersize=10)
    ax_compare=embedding_a[23:27,0]
    ay_compare=embedding_a[23:27,1]
    plt.plot(ax_compare,ay_compare, 'o',color='y',markersize=10)
    ax_compare=embedding_a[27:30,0]
    ay_compare=embedding_a[27:30,1]
    plt.plot(ax_compare,ay_compare, 'o',color='c',markersize=10)
    # ax_compare=embedding_a[28:31,0]
    # ay_compare=embedding_a[28:31,1]
    # plt.plot(ax_compare,ay_compare, 'o',color='b',markersize=5)
    # ax_compare=embedding_a[35:40,0]
    # ay_compare=embedding_a[35:40,1]
    # plt.plot(ax_compare,ay_compare, 'o',color='gold',markersize=5)
    # ax_compare=embedding_a[40:44,0]
    # ay_compare=embedding_a[40:44,1]
    # plt.plot(ax_compare,ay_compare, 'o',color='grey',markersize=5)
    # ax_compare=embedding_a[44:48,0]
    # ay_compare=embedding_a[44:48,1]
    # plt.plot(ax_compare,ay_compare, 'o',color='pink',markersize=5)


    bx_compare=embedding_b[0:5,0]
    by_compare=embedding_b[0:5,1]
    plt.plot(bx_compare,by_compare, '^',color='r',markersize=10)
    bx_compare=embedding_b[5:8,0]
    by_compare=embedding_b[5:8,1]
    plt.plot(bx_compare,by_compare, '^',color='k',markersize=10)
    bx_compare=embedding_b[8:13,0]
    by_compare=embedding_b[8:13,1]
    plt.plot(bx_compare,by_compare, '^',color='g',markersize=10)
    bx_compare=embedding_b[13:23,0]
    by_compare=embedding_b[13:23,1]
    plt.plot(bx_compare,by_compare, '^',color='m',markersize=10)
    bx_compare=embedding_b[23:27,0]
    by_compare=embedding_b[23:27,1]
    plt.plot(bx_compare,by_compare, '^',color='y',markersize=10)
    bx_compare=embedding_b[27:30,0]
    by_compare=embedding_b[27:30,1]
    plt.plot(bx_compare,by_compare, '^',color='c',markersize=10)
    # bx_compare=embedding_b[28:31,0]
    # by_compare=embedding_b[28:31,1]
    # plt.plot(bx_compare,by_compare, '^',color='b',markersize=5)
    # bx_compare=embedding_b[35:40,0]
    # by_compare=embedding_b[35:40,1]
    # plt.plot(bx_compare,by_compare, '^',color='gold',markersize=5)
    # bx_compare=embedding_b[40:44,0]
    # by_compare=embedding_b[40:44,1]
    # plt.plot(bx_compare,by_compare, '^',color='grey',markersize=5)
    # bx_compare=embedding_b[44:48,0]
    # by_compare=embedding_b[44:48,1]
    # plt.plot(bx_compare,by_compare, '^',color='pink',markersize=5)
    plt.savefig('./eval.jpg')

        





if __name__ == '__main__':
    args = parse_args()

    # 数据处理过程
    with open("./archive/eval.txt", "r") as f:
        data = f.readlines()
        image_list=[]
        for i in range(1,len(data)):
            temp=data[i].split(".jpg")
            image_list.append(temp[0]+".jpg")

        caption_list=[]
        for i in range(1,len(data)):
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