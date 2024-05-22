import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import pylab


device = torch.device('cuda:7' if torch.cuda.is_available else 'cpu')


def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if torch.cuda.is_available():
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas).to(device))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value



def mmd_loss_gpt(x, y, kernel_type='rbf', kernel_bandwidth=1.0):
    # 计算两个输入数据的样本数量
    m = x.size(0)
    n = y.size(0)

    # 计算核矩阵
    xx = torch.matmul(x, x.t()).to(device)
    yy = torch.matmul(y, y.t()).to(device)
    xy = torch.matmul(x, y.t()).to(device)

    if kernel_type == 'rbf':
        # 使用 RBF 核函数计算核矩阵的元素
        k_xx = torch.exp(-torch.div(torch.pow(torch.norm(x.unsqueeze(1) - x, dim=2), 2), kernel_bandwidth)).to(device)
        k_yy = torch.exp(-torch.div(torch.pow(torch.norm(y.unsqueeze(1) - y, dim=2), 2), kernel_bandwidth)).to(device)
        k_xy = torch.exp(-torch.div(torch.pow(torch.norm(x.unsqueeze(1) - y, dim=2), 2), kernel_bandwidth)).to(device)
    else:
        raise ValueError("Unsupported kernel type. Please choose 'rbf'.")

    # 计算 MMD 损失
    loss = torch.mean(k_xx).to(device) - 2 * torch.mean(k_xy).to(device) + torch.mean(k_yy).to(device)

    return loss.to(device)
