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



if __name__ == '__main__':
    data=[]
    with open("./archive/captions1.txt", "r") as f:
        data = f.readlines()
        # print(data)
    with open('./archive/captions.txt','w') as f1: 
        f1.write(data[0])   
        for i in range(1,len(data)):
            if i%5==1 or i%5==4:
                f1.write(data[i])

    pass