from torch import nn
import torch
# 这里面存有预训练好的模型
import torchvision.models as models


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        # 暂时定义的resnet34
        resnet34 = models.resnet34(pretrained=True)
        # children()和modules()的不同,https://blog.csdn.net/LXX516/article/details/79016980
        self.feature_layers = nn.Sequential(*list(resnet34.children())[:-1])
        # self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        x = self.feature_layers(input)
        output = x.view(x.size(0), -1)
        # output = self.fc(x)
        
        return output


if __name__ == "__main__":
    net=Resnet34()
    print(net)
    pass