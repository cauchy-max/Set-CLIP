import torch
from torch import nn
import torch.nn.functional as F

from ResNet import *
from vit import *
from ShuffleNet import *
from SimCSE import *
from Collator import *
from mkmmd_loss import *

# Bert-Tokenizer不算是模型中的部分，不需要反向传播
# gpu or cpu    数据，模型，损失函数 主要是其中的torch定义的tensor，优化函数
# 是否需要梯度,model.train()/model.eval()
# 损失值大有优势，这样在反向传播中能起更大作用。如果两个损失对抗，按某个梯度，一个减小100多，一个增长0.2，肯定会选这个梯度。

device = torch.device('cuda:7' if torch.cuda.is_available else 'cpu')

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x




class CLIP_plus(nn.Module):
    def __init__(
        self,
        image_embedding=512,
        text_embedding=768,
    ):
        super().__init__()
        self.image_encoder = Resnet34()
        self.text_encoder = SimCSE()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        # self.temperature = temperature
        # self.map_layer = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.Sigmoid(),
        #     nn.Linear(512, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 8),
        #     nn.Sigmoid(),
        #     nn.Linear(8, 2))

    def forward(self, batch):
        # 这个batch里包含了真实数据和增强后的数据
        # Getting Image and Text Features
        image=[]
        input_ids=[]
        attention_mask=[]
        token_type_ids=[]

        for i in range(len(batch)):
            image.append(batch[i]['image'].cpu().numpy().tolist()) 
            input_ids.append(batch[i]['input_ids'].cpu().numpy().tolist()) 
            attention_mask.append(batch[i]['attention_mask'].cpu().numpy().tolist()) 
            token_type_ids.append(batch[i]['token_type_ids'].cpu().numpy().tolist())
        image1=torch.Tensor(image).to(device) 
        # image2=torch.Tensor(image).to(device)
        input_ids1=torch.Tensor(input_ids).to(torch.int).to(device)
        attention_mask1=torch.Tensor(attention_mask).to(device)
        token_type_ids1=torch.Tensor(token_type_ids).to(torch.int).to(device)



        image_features = self.image_encoder(image1)
        text_features = self.text_encoder(
            input_ids=input_ids1, attention_mask=attention_mask1,token_type_ids=token_type_ids1
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # image_map = self.map_layer(image_embeddings)
        # text_map = self.map_layer(text_embeddings)

        image_norm = F.normalize(image_embeddings, p=2, dim=1)
        text_norm = F.normalize(text_embeddings, p=2, dim=1)

        return image_norm,text_norm



class CLIP_plus_vit(nn.Module):
    def __init__(
        self,
        image_embedding=768,
        text_embedding=768,
    ):
        super().__init__()
        self.image_encoder = vit()
        self.text_encoder = SimCSE()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        # self.temperature = temperature
        # self.map_layer = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.Sigmoid(),
        #     nn.Linear(512, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 8),
        #     nn.Sigmoid(),
        #     nn.Linear(8, 2))

    def forward(self, batch):
        # 这个batch里包含了真实数据和增强后的数据
        # Getting Image and Text Features
        image=[]
        input_ids=[]
        attention_mask=[]
        token_type_ids=[]

        for i in range(len(batch)):
            image.append(batch[i]['image'].cpu().numpy().tolist()) 
            input_ids.append(batch[i]['input_ids'].cpu().numpy().tolist()) 
            attention_mask.append(batch[i]['attention_mask'].cpu().numpy().tolist()) 
            token_type_ids.append(batch[i]['token_type_ids'].cpu().numpy().tolist())
        image1=torch.Tensor(image).to(device) 
        # image2=torch.Tensor(image).to(device)
        input_ids1=torch.Tensor(input_ids).to(torch.int).to(device)
        attention_mask1=torch.Tensor(attention_mask).to(device)
        token_type_ids1=torch.Tensor(token_type_ids).to(torch.int).to(device)



        image_features = self.image_encoder(image1)
        text_features = self.text_encoder(
            input_ids=input_ids1, attention_mask=attention_mask1,token_type_ids=token_type_ids1
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # image_map = self.map_layer(image_embeddings)
        # text_map = self.map_layer(text_embeddings)

        image_norm = F.normalize(image_embeddings, p=2, dim=1)
        text_norm = F.normalize(text_embeddings, p=2, dim=1)

        return image_norm,text_norm



if __name__ == '__main__':
    
    # 这一部分已经不对了，具体参考dataset.py的形式，他是[{},{},{}]的形式，而不是{[],[],[]}
    images = torch.randn(8, 3, 224, 224).to(device)
    print(images)
    input_ids = torch.randint(5, 300, size=(8, 25)).to(device)
    # print(input_ids)
    attention_mask = torch.ones(8, 25).to(device)
    token_type_ids = torch.zeros(8,25)
    token_type_ids = token_type_ids.to(torch.int).to(device)
    # print(token_type_ids)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


    CLIP = CLIP_plus().to(device)
    a,b = CLIP(batch)
    print(b.shape)
    print(b[0::2].shape)
    mmd_loss = mmd_loss(a, b) * 0.5 + \
               mmd_loss(a, b) * 0.5
    print(mmd_loss)
