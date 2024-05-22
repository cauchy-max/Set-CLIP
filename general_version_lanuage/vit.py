import torch.nn as nn
from transformers import AutoImageProcessor, ViTModel
import torch


class vit(nn.Module):
    def __init__(self):
        super(vit, self).__init__()
        self.vit_model = ViTModel.from_pretrained("./vit_pretrained_weight")
        
    def forward(self, input):
        # 获取模型输出
        outputs = self.vit_model(pixel_values=input)

        # 从模型输出中提取嵌入
        # 使用分类标记作为整个图像的嵌入
        image_embedding = outputs.last_hidden_state[:, 0]
        return image_embedding

if __name__ == "__main__":
    net=vit()
    print(net)
    pass