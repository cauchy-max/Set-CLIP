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
from transformers import BertConfig, BertModel, BertTokenizer, AutoImageProcessor, ViTModel

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
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=31, help="epochs")
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
    dataset = CLIPDataset_vit(
        image_array,
        caption_array,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    collator = Collator(tokenizer)
    # 每一个dataloader里面有2*batch_size个数据
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collator.collate_semantic
    )
    return dataloader




def finetune(args,image_array,caption_array,mode="train"):
    # tokenizer在这里定义
    tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")
    # mode在build_loaders里面才用
    dl=build_loaders(args,image_array,caption_array,tokenizer,mode)
    
    # model = torch.load('model_unsupervised_49_2000.pkl')
    model = CLIP_plus_vit().to(device)   
    optimizer = torch.optim.Adam([
        {'params': model.image_encoder.parameters()},
        {'params': model.text_encoder.parameters()},
        {'params': model.image_projection.parameters(), 'lr': args.lr},
        {'params': model.text_projection.parameters(), 'lr': args.lr}], 
        lr=args.lr/100, weight_decay=1e-3
    )
    # 更新学习率
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.8, verbose=True
    )

    model.train()
    model.to(device)
    figure_list1=[]
    flag1=1

    for epoch_idx in range(args.epochs):
        loss_epoch=0
        print("========="+str(epoch_idx)+"===========")
        for batch in tqdm(dl):
            image_embeddings,text_embeddings = model(batch)
            ########################################
            if(epoch_idx==args.epochs-1) and flag1==1:
                figure_list1=[]
                # print(image_pred.shape)
                figure_list1.append(image_embeddings)
                figure_list1.append(text_embeddings)
                flag1=1
            ########################################
            # Calculating the Loss
            temperature = 0.05
            # 因为做了normalize，所以这就相当于cosine similarity了
            logits = (text_embeddings @ image_embeddings.T) / temperature
            # images_similarity = image_embeddings @ image_embeddings.T
            # texts_similarity = text_embeddings @ text_embeddings.T
            targets = torch.arange(0, image_embeddings.shape[0], device=device)
            # targets = F.softmax(
            #     (images_similarity + texts_similarity) / 2 * temperature, dim=-1
            # )
            texts_loss = F.cross_entropy(logits, targets)
            images_loss = F.cross_entropy(logits.T, targets)
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            loss_clip = loss.mean()

            loss = loss_clip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch=loss.item()
        
        print(str(epoch_idx)+":  "+str(loss_epoch))
        lr_scheduler.step(loss_epoch)

        if(epoch_idx==29):
            # 模型保存
            torch.save(model, 'vit_clip_'+str(epoch_idx)+'_15000.pkl')

    
    # 画图了
    a=figure_list1[0]
    # print(a)
    b=figure_list1[1]
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    # print(a.shape)
    reducer = umap.UMAP(n_neighbors=7, n_components=2)
    whole=np.concatenate((a,b),axis=0)
    reducer=reducer.fit(whole)
    # 使用UMAP进行降维
    embedding_a = reducer.transform(a)
    embedding_b = reducer.transform(b)

    fig1 = plt.figure(figsize=(4, 3))
    ax=embedding_a[:,0]
    ay=embedding_a[:,1]
    plt.plot(ax,ay, 'o',color='b',markersize=5)
    ax_compare=embedding_a[0:4,0]
    ay_compare=embedding_a[0:4,1]
    plt.plot(ax_compare,ay_compare, 'o',color='r',markersize=5)
    # ax_compare=embedding_a[12:18,0]
    # ay_compare=embedding_a[12:18,1]
    # plt.plot(ax_compare,ay_compare, 'o',color='m',markersize=5)

    bx=embedding_b[:,0]
    by=embedding_b[:,1]
    plt.plot(bx,by, '^',color='g',markersize=5)
    bx_compare=embedding_b[0:4,0]
    by_compare=embedding_b[0:4,1]
    plt.plot(bx_compare,by_compare, '^',color='y',markersize=5)
    # bx_compare=embedding_b[12:18,0]
    # by_compare=embedding_b[12:18,1]
    # plt.plot(bx_compare,by_compare, '^',color='pink',markersize=5)
    plt.savefig('./finetune.jpg')





if __name__ == '__main__':
    args = parse_args()

    import json
    with open('./mscoco/captions.json') as f:
        superHeroSquad = json.load(f)
        # print(superHeroSquad[0])
        # print(superHeroSquad[1])
        image_list=[]
        for i in range(0,15000):
            temp=str(superHeroSquad[i]['image_id'])
            rest_len=12-len(temp)
            before='0'*rest_len
            before='COCO_train2014_'+before
            temp=before+temp+'.jpg'
            image_list.append(temp)
        
        caption_list=[]
        for i in range(0,15000):#len(data)):
            temp=superHeroSquad[i]['caption']
            caption_list.append(temp.replace("\n",""))
    
    
    
    
    # # 数据处理过程
    # with open("./archive/captions.txt", "r") as f:
    #     data = f.readlines()
    #     image_list=[]
    #     for i in range(1,6001):#len(data)):
    #         temp=data[i].split(".jpg")
    #         image_list.append(temp[0]+".jpg")

    #     caption_list=[]
    #     for i in range(1,6001):#len(data)):
    #         temp=data[i].split(".jpg,")
    #         caption_list.append(temp[1].replace("\n",""))
            
            
            
            


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


        finetune(args,image_array,caption_array,mode="train")