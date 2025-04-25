'''
Author: 星必尘Sguan
Date: 2025-04-24 21:31:29
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 15:41:36
FilePath: \test_TrainPytorch\nn_maxpool.py
Description: 池化神经网络的学习MaxPool2d

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import MaxPool2d
import torchvision

# 1测试的数据集
dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

# 2自己创建的图像数据
img_input = torch.tensor([[1,2,0,3,1],
                          [0,1,2,3,1],
                          [1,2,1,0,0],
                          [5,2,3,1,1],
                          [2,1,0,1,1]])
# 2图片重定义reshap
img_input = torch.reshape(img_input,(1,1,5,5)) 

# 池化层 神经网络搭建
class Sguan(nn.Module):
    def __init__(self):
        super(Sguan,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        x = self.maxpool1(x)
        return x

# 2自己创建数据集的计算
sguan_maxpool = Sguan()
output = sguan_maxpool(img_input)
print(output)

# 1测试数据集的显示
Step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs,targets = data
    img_out = sguan_maxpool(imgs)
    writer.add_images("Maxpool_in",imgs,Step)
    writer.add_images("Maxpool_out",img_out,Step)
    Step += 1

writer.close()

