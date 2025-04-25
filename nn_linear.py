'''
Author: 星必尘Sguan
Date: 2025-04-25 13:52:43
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 14:16:47
FilePath: \test_TrainPytorch\nn_linear.py
Description: 神经网络线性层Linear_Layers的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear
import torchvision

# 测试的数据集
dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)



class Sguan(nn.Module):
    def __init__(self):
        super(Sguan,self).__init__()
        self.linear1 = Linear(in_features=196608,out_features=10)

    def forward(self,x):
        x = self.linear1(x)
        return x

sguan_linear = Sguan()

for data in dataloader:
    imgs,targets = data
    # 方法一 使用重定义尺寸函数
    # 此处reshape参数的"-1"是指的"开启自适应模式"
    # img_output = torch.reshape(imgs,(1,1,1,-1))

    # 方法二 直接使用flatten函数展平图片
    img_output = torch.flatten(imgs) # flatten函数可以把图片展平
    print(img_output.shape)
    img_output = sguan_linear(img_output)
    print(img_output.shape)

