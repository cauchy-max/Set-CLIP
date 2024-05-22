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


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]




def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("train_file", type=str, help="train text file")
    # parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    # parser.add_argument("--model_out", type=str, default="./model", help="model output path")
    # parser.add_argument("--num_proc", type=int, default=5, help="dataset process thread num")
    # parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=32, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    # 数据排布的话，一个dataloader里需要前sup_num个一一对应，后一些没有一一对应关系，按照同一分布随便采集
    # 这块放在数据集预处理部分。比方说定义batch size为32，则数据集则每32个前16个为对应的图文，后16个则取不对应的图文
    # 这样模型整体不用变，只需要修改输入的数据集内容即可
    parser.add_argument("--sup_num", type=float, default=21, help="supervised sample number in one epoch")
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
        # 丢弃最后一个不满的dataloader，防止报错
        drop_last = True,
        collate_fn=collator.collate_vit
    )
    return dataloader



# 这个y_pred是增强后的数据送入模型得到的结果，但不是全部结果，是只有图片的embeddings
def compute_loss_image(y_pred, tao=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # # similarity转为0-1之间
    # similarities=(similarities+1)/2
    # # 将对角线元素置为0，使其相似度为0，方便后面的归一化
    # diagonal = torch.diag(torch.diag(similarities))
    # similarities = similarities - diagonal
    # # 归一化
    # row_sums = similarities.sum(dim=1, keepdim=True)
    # similarities = similarities / row_sums

    # 将相似性矩阵（similarities）的对角线元素设置为负无穷大。避免与自身比较
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    # print(similarities)
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss).to(device)



# 这个y_pred是增强后的数据送入模型得到的结果
def compute_loss_text(y_pred, tao=0.05):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    # # similarity转为0-1之间
    # similarities=(similarities+1)/2
    # # 将对角线元素置为0，使其相似度为0，方便后面的归一化
    # diagonal = torch.diag(torch.diag(similarities))
    # similarities = similarities - diagonal
    # # 归一化
    # row_sums = similarities.sum(dim=1, keepdim=True)
    # similarities = similarities / row_sums

    # 将相似性矩阵（similarities）的对角线元素设置为负无穷大。避免与自身比较
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss).to(device)




# # 创建KernelDensity对象
# kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
# kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5)

# # 定义自定义损失函数，核密度函数，用于估计分布密度
# class KDELoss(nn.Module):
#     def __init__(self, kde, kde1):
#         super(KDELoss, self).__init__()
#         self.kde = kde
#         self.kde1 = kde1

#     def forward(self, input_image,input_text):
#         # 将输入数据转换为numpy数组
#         input_image_np = input_image.detach().cpu().numpy()
#         self.kde.fit(input_image_np)
#         input_text_np = input_text.detach().cpu().numpy()
#         self.kde1.fit(input_text_np)

#         # 展平输入数据为[batch_size, 256]的形状
#         input_image_flat = input_image_np.reshape(input_image_np.shape[0], -1)
#         input_text_flat = input_text_np.reshape(input_text_np.shape[0], -1)

#         # 计算对数密度
#         log_density_text = self.kde.score_samples(input_text_flat)
#         log_density_image = self.kde1.score_samples(input_text_flat)

#         log_density_text=torch.tensor(log_density_text).requires_grad_(True).to(device)
#         log_density_image=torch.tensor(log_density_image).requires_grad_(True).to(device)
        

#         # log_density=[x + y for x, y in zip(log_density_text, log_density_image)]
        
#         loss_mse = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
#         loss = loss_mse(log_density_image, log_density_text)

#         # 计算损失（对数密度的负平均）
#         # loss = -torch.mean(torch.tensor(log_density_text))

#         return loss.to(device)


class DifferentiableKernelDensityEstimation(nn.Module):
    def __init__(self, bandwidth=1.0):
        super(DifferentiableKernelDensityEstimation, self).__init__()
        self.bandwidth = bandwidth

    def forward(self, x, data):
        # 计算 Parzen 窗函数的可导版本
        diff_kernel = -((x.unsqueeze(1) - data) / self.bandwidth).pow(2) / 2
        diff_kernel = torch.exp(diff_kernel.sum(dim=2) / (self.bandwidth * (2 * 3.14159265358979323846)**0.5)).to(device)

        # 计算密度估计
        density = diff_kernel.mean(dim=1, keepdim=True)

        return density.to(device)



def gaussian_kernel(x, y, bandwidth):
    """计算高斯核，适用于多维数据."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # 扩展维度以便于广播
    dist_sq = torch.sum(diff.pow(2), dim=-1)  # 计算平方欧氏距离
    return torch.exp(-0.5 * (dist_sq / bandwidth.pow(2)))

def kde(x, samples, bandwidth):
    """多维核密度估计."""
    kernel_values = gaussian_kernel(x, samples, bandwidth)
    return kernel_values.mean(dim=1)




# torch.pi = math.pi
# def gaussian_kernel_density_estimate(x, data, bandwidth):
#     # 计算样本与数据集之间的欧氏距离平方
#     distance = torch.sum((data - x) ** 2, dim=1)
#     # 计算高斯核密度估计的分子部分
#     numerator = torch.exp(-distance / (2 * bandwidth ** 2))
#     # 计算高斯核密度估计的分母部分
#     denominator = torch.sqrt(2 * torch.tensor(torch.pi)) * bandwidth
#     # 计算核密度估计值
#     density_estimate = torch.mean(numerator / denominator)
#     return density_estimate


# def compare_density_estimates(data_A, data_B, bandwidth=0.5):
#     # 初始化核密度估计比较结果
#     result = torch.zeros(data_A.size(0))
#     # 对数据集A中的每个样本进行核密度估计比较
#     for i, x in enumerate(data_A):
#         # 对样本x在数据集B中进行核密度估计
#         density_estimate_B = gaussian_kernel_density_estimate(x, data_B, bandwidth)
#         # 对样本x在数据集A中进行核密度估计
#         density_estimate_A = gaussian_kernel_density_estimate(x, data_A, bandwidth)
#         # 计算核密度估计比较结果
#         result[i] = density_estimate_A / density_estimate_B
#     return result




def gaussian_kernel_density_estimate(x, data, bandwidth):
    num_samples = data.size(0)
    num_features = data.size(1)

    variance_vec = torch.var(data, dim=0)
    variance = torch.sum(variance_vec)
    # 计算样本与数据集之间的欧氏距离平方,后面除以方差得到相对距离吧。
    distance_temp = torch.sum((data - x) ** 2, dim=1)#可以sqrt，也可以除一个大数
    distance = distance_temp/variance
    # 计算高斯核密度估计的分子部分,*3用于放大
    numerator = 3*(torch.exp(-distance / (4 * bandwidth ** 2))-0.0)#这里正常distance部分前为-
    # print(numerator)
    # 计算高斯核密度估计的分母部分
    denominator = (2 * math.pi * bandwidth ** 2)#** (num_features / 2)
    # 计算核密度估计值
    density_estimate = torch.sum(numerator / denominator)/16
    # print(density_estimate)
    return density_estimate


def compare_density_estimates(data_A, data_B, bandwidth=0.5):
    num_samples_A = data_A.size(0)
    num_samples_B = data_B.size(0)

    # 初始化核密度估计比较结果
    result = torch.zeros(num_samples_A)
    # 对数据集A中的每个样本进行核密度估计比较
    for i in range(num_samples_A):
        x = data_A[i]
        # 对样本x在数据集B中进行核密度估计
        density_estimate_B = gaussian_kernel_density_estimate(x, data_B, bandwidth)
        # 对样本x在数据集A中进行核密度估计
        density_estimate_A = gaussian_kernel_density_estimate(x, data_A, bandwidth)
        # 计算核密度估计比较结果
        result[i] = density_estimate_A / density_estimate_B

    # print(result)
    # print("======")
    return result


# 用于计算簇间的真实差异而非比值
def compare_density_for_cluster(data_A, data_B, bandwidth=0.5):
    num_samples_A = data_A.size(0)
    num_samples_B = data_B.size(0)

    # 初始化核密度估计比较结果
    result = torch.zeros(num_samples_A)
    # 对数据集A中的每个样本进行核密度估计比较
    for i in range(num_samples_A):
        x = data_A[i]
        # 对样本x在数据集B中进行核密度估计
        density_estimate_B = gaussian_kernel_density_estimate(x, data_B, bandwidth)
        # 对样本x在数据集A中进行核密度估计
        density_estimate_A = gaussian_kernel_density_estimate(x, data_A, bandwidth)
        # 计算核密度估计比较结果
        # 这一块是直接用减法做
        result[i] = (density_estimate_A - density_estimate_B)**2

    dis=sum(result)
    # print(dis)
    # print("======")
    return dis




def two_way_divergence(p, q):
    divergence = torch.sum(p * torch.log(p / q)) + torch.sum(q * torch.log(q / p))
    return divergence

# 用于计算簇间的真实差异而非比值
def compare_density_for_cluster_with_kl(data_A, data_B, bandwidth=0.5):
    num_samples_A = data_A.size(0)
    num_samples_B = data_B.size(0)

    # 初始化核密度估计比较结果
    resultA = torch.zeros(num_samples_A)
    resultB = torch.zeros(num_samples_B)
    # 对数据集A中的每个样本进行核密度估计比较
    for i in range(num_samples_A):
        x = data_A[i]
        # 对样本x在数据集B中进行核密度估计
        density_estimate_B = gaussian_kernel_density_estimate(x, data_B, bandwidth)
        # 对样本x在数据集A中进行核密度估计
        density_estimate_A = gaussian_kernel_density_estimate(x, data_A, bandwidth)
        # 计算核密度估计比较结果
        resultA[i] = density_estimate_A
        resultB[i] = density_estimate_B

    # sumA=sum(resultA)
    # sumB=sum(resultB)
    # for i in range(len(resultA)):
    #     resultA[i] = resultA[i]/sumA
    #     resultB[i] = resultB[i]/sumB
    divergence = two_way_divergence(resultA, resultB)
    # print(divergence)
    # print(result)
    # print("======")
    return divergence




def train(args,image_array,caption_array,mode="train"):
    # tokenizer在这里定义
    tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")
    # mode在build_loaders里面才用
    dl=build_loaders(args,image_array,caption_array,tokenizer,mode)
    model = CLIP_plus_vit().to(device)
    # print(model)
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
    sup_num = args.sup_num


    model.train()
    # ###############
    figure_list1=[]
    figure_list2=[]
    flag1=1
    flag2=1
    #####################
    for epoch_idx in range(args.epochs):
        loss_epoch=0
        loss_temp=0
        loss_temp1=0
        loss_temp2=0
        print("========="+str(epoch_idx)+"===========")
        for batch in tqdm(dl):
            image_embeddings,text_embeddings = model(batch)
            ########################################
            if(epoch_idx==args.epochs-2) and flag1==1:
                figure_list1=[]
                # print(image_pred.shape)
                figure_list1.append(image_embeddings)
                figure_list1.append(text_embeddings)
                flag1=1
            if(epoch_idx==args.epochs-1) and flag2==1:
                figure_list2=[]
                figure_list2.append(image_embeddings)
                figure_list2.append(text_embeddings)
                flag2=1
            ########################################
            loss_image=compute_loss_image(image_embeddings)
            loss_text=compute_loss_text(text_embeddings)
            loss_mmd = mmd_loss_gpt(image_embeddings[0::2], text_embeddings[0::2]) 

            average_image=torch.mean(image_embeddings[0::2],dim=0,keepdim=True).to(device)
            average_text=torch.mean(text_embeddings[0::2],dim=0,keepdim=True).to(device)
            loss_mse = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss_average = loss_mse(average_image, average_text)

            
            
            # 保证大簇而小差
            average_distance=torch.sum((average_image - average_text) ** 2)
            image_max=torch.max(image_embeddings[0::2],dim=0,keepdim=True)[0].to(device)
            image_min=torch.min(image_embeddings[0::2],dim=0,keepdim=True)[0].to(device)
            image_distance=torch.sum((image_max - image_min) ** 2)
            # 用方差应该更好
            image_mean = torch.mean(image_embeddings[0::2], dim=0)
            image_diff = image_embeddings[0::2] - image_mean
            image_variance = torch.var(image_diff, dim=0)
            image_variance = torch.sum(image_variance)
            

            text_max=torch.max(text_embeddings[0::2],dim=0,keepdim=True)[0].to(device)
            text_min=torch.min(text_embeddings[0::2],dim=0,keepdim=True)[0].to(device)
            text_distance=torch.sum((text_max - text_min) ** 2)
            # 用方差应该更好
            text_mean = torch.mean(text_embeddings[0::2], dim=0)
            text_diff = text_embeddings[0::2] - text_mean
            text_variance = torch.var(text_diff, dim=0)
            text_variance = torch.sum(text_variance)
            # print(text_distance)
            # print(text_variance)
            # print("========")
            # loss_cluster=3*(torch.exp(2*average_distance)-1)/image_distance+3*(torch.exp(2*average_distance)-1)/text_distance

            dis = compare_density_for_cluster_with_kl(image_embeddings[0::2], text_embeddings[0::2], bandwidth=0.5)
            dis1 = compare_density_for_cluster_with_kl(text_embeddings[0::2], image_embeddings[0::2], bandwidth=0.5)
            loss_cluster=(dis+dis1)/image_distance+(dis+dis1)/text_distance
            # loss_cluster=(dis+dis1)/image_variance+(dis+dis1)/text_variance



            # 直接加入有监督约束
            temperature = 0.05
            # 因为做了normalize，所以这就相当于cosine similarity了
            logits = (text_embeddings[0::2] @ image_embeddings[0::2].T) / temperature
            # images_similarity = image_embeddings @ image_embeddings.T
            # texts_similarity = text_embeddings @ text_embeddings.T
            targets = torch.arange(0, sup_num, device=device)
            # targets = F.softmax(
            #     (images_similarity + texts_similarity) / 2 * temperature, dim=-1
            # )
            texts_loss = F.cross_entropy(logits[0:sup_num], targets)
            images_loss = F.cross_entropy(logits.T[0:sup_num], targets)
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            loss_clip = loss.mean()


            
            
            # 创建损失函数对象
            # kde_model_text = DifferentiableKernelDensityEstimation()
            # density_estimate_text = kde_model_text(text_pred[0::2], text_pred[0::2]).view(1, -1)
            # density_estimate_text1 = kde_model_text(image_pred[0::2], text_pred[0::2]).view(1, -1)
            # kde_model_image = DifferentiableKernelDensityEstimation()
            # density_estimate_image = kde_model_image(text_pred[0::2], image_pred[0::2]).view(1, -1)
            # density_estimate_image1 = kde_model_image(image_pred[0::2], image_pred[0::2]).view(1, -1)


            # # bandwidth = torch.tensor(1.0)  # 核带宽
            # # density_estimate_text = kde(text_pred[0::2], text_pred[0::2], bandwidth).view(1, -1)
            # # density_estimate_image = kde(text_pred[0::2], image_pred[0::2], bandwidth).view(1, -1)


            # loss_MSE = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            # loss_kde = loss_MSE(density_estimate_image, density_estimate_text)
            # loss_kde1 = loss_MSE(density_estimate_image1, density_estimate_text1)
            # # print(loss_kde)
            
            
            density_ratios = compare_density_estimates(image_embeddings[0::2], text_embeddings[0::2], bandwidth=0.5)
            # target = torch.ones(image_pred[0::2].size(0))
            loss_MSE = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss_kde = loss_MSE(density_ratios, torch.ones_like(density_ratios))
                        


            loss=0.1*loss_image+0.1*loss_text+0.6*loss_cluster+0.1*loss_mmd+0*loss_average+1.0*loss_clip#+0.5*loss_kde+0.0005*loss_text+loss_kde1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch=loss.item()
            loss_temp=loss_cluster.item()
            loss_temp1=loss_image.item()
            loss_temp2=loss_clip.item()
        
        print(str(epoch_idx)+":  "+str(loss_epoch))
        print(str(epoch_idx)+":  "+str(loss_temp))
        print(str(epoch_idx)+":  "+str(loss_temp1))
        print(str(epoch_idx)+":  "+str(loss_temp2))
        lr_scheduler.step(loss_epoch)

        if(epoch_idx==29):
            # 模型保存
            torch.save(model, 'vit_end2end_'+str(epoch_idx+1)+'_15000_5000_improve_new_relative_distance_kl.pkl')

    # 画图了
    a=figure_list1[0]
    # print(a)
    b=figure_list1[1]
    c=figure_list2[0]
    d=figure_list2[1]
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    c = c.detach().cpu().numpy()
    d = d.detach().cpu().numpy()
    # print(a.shape)

    # embedding_a=a
    # embedding_b=b
    # embedding_c=c
    # embedding_d=d
    # print(embedding_a.shape)
    reducer = umap.UMAP(n_neighbors=7, n_components=2)
    whole=np.concatenate((a,b,c,d),axis=0)
    reducer=reducer.fit(whole)
    # 使用UMAP进行降维
    embedding_a = reducer.transform(a)
    embedding_b = reducer.transform(b)
    embedding_c = reducer.transform(c)
    embedding_d = reducer.transform(d)
    # print(embedding_a.shape)

    fig1 = plt.figure(figsize=(4, 3))
    ax=embedding_a[:,0]
    ay=embedding_a[:,1]
    plt.plot(ax,ay, 'o',color='b')
    plt.savefig('./image_1.jpg')
    bx=embedding_b[:,0]
    by=embedding_b[:,1]
    plt.plot(bx,by, 'o',color='g')
    plt.savefig('./text_1.jpg')

    fig2 = plt.figure(figsize=(4, 3))
    cx=embedding_c[:,0]
    cy=embedding_c[:,1]
    plt.plot(cx,cy, 'o',color='b',label='image')
    cx_compare=embedding_c[0:4,0]
    cy_compare=embedding_c[0:4,1]
    plt.plot(cx_compare,cy_compare, 'o',color='r')
    plt.savefig('./image_n.jpg')

    dx=embedding_d[:,0]
    dy=embedding_d[:,1]
    plt.plot(dx,dy, 'o',color='g',label='text')
    dx_compare=embedding_d[0:4,0]
    dy_compare=embedding_d[0:4,1]
    plt.plot(dx_compare,dy_compare, 'o',color='y')
    plt.legend()
    plt.savefig('./text_n.jpg')


    pass





if __name__ == '__main__':
    args = parse_args()

    import json
    with open('./mscoco/captions.json') as f:
        superHeroSquad = json.load(f)
        # print(superHeroSquad[0])
        # print(superHeroSquad[1])
        image_list=[]
        for i in range(0,15040):
            temp=str(superHeroSquad[i]['image_id'])
            rest_len=12-len(temp)
            before='0'*rest_len
            before='COCO_train2014_'+before
            temp=before+temp+'.jpg'
            image_list.append(temp)
        
        caption_list=[]
        for i in range(0,15040):#len(data)):
            temp=superHeroSquad[i]['caption']
            caption_list.append(temp.replace("\n",""))
    
    
    
    # # 数据处理过程
    # with open("./archive/captions.txt", "r") as f:
    #     data = f.readlines()
    #     image_list=[]
    #     for i in range(1,6017):#len(data)):
    #         temp=data[i].split(".jpg")
    #         image_list.append(temp[0]+".jpg")

    #     caption_list=[]
    #     for i in range(1,6017):#len(data)):
    #         temp=data[i].split(".jpg,")
    #         caption_list.append(temp[1].replace("\n",""))



        # # 有标签的多训练几遍，将一个batch中的有标签数据增加，防止batch中有标签的约束与引导太少，导致走偏
        # image_list_with_label = image_list[0:2000]
        # image_list_with_label = image_list_with_label+image_list_with_label
        # image_list_without_label = image_list[2000:6000]
        # image_list_last=[]
        # for i in range(int(len(image_list_with_label)/32)):
        #     image_list_last = image_list_last+image_list_with_label[32*i:32*(i+1)]+image_list_without_label[32*i:32*(i+1)]
        #     pass
        
        # caption_list_with_label = caption_list[0:2000]
        # caption_list_with_label = caption_list_with_label+caption_list_with_label
        # caption_list_without_label = caption_list[2000:6000]
        # caption_list_last=[]
        # for i in range(int(len(caption_list_with_label)/32)):
        #     caption_list_last = caption_list_last+caption_list_with_label[32*i:32*(i+1)]+caption_list_without_label[32*i:32*(i+1)]
        #     pass
        
        # print(len(caption_list_last))
        # # 3968为64*62，正常结束，3969~3984为with_label的最后16个，3985~4000为第二遍with_label的前16个
        # # 4000~4032为without_label的第63个
        # print(caption_list_last[0])
        # print(caption_list_last[3984])
        # print(image_list_last[3984])






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


        train(args,image_array,caption_array,mode="train")

        
        
        
        # # 测试dataset
        # # transforms = get_transforms(mode="train")
        # tokenizer = BertTokenizer.from_pretrained("./simcse_pretrained_weight")


        # dl=build_loaders(args,image_array,caption_array,tokenizer,"train")
        # # for batch in tqdm(dl):
        # #     print(batch)
        # print(len(list(dl)[0]))
        # print("1")
        # print("2")
    pass

