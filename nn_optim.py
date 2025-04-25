'''
Author: 星必尘Sguan
Date: 2025-04-25 17:51:15
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 19:12:25
FilePath: \test_TrainPytorch\nn_optim.py
Description: 神经网络优化器Optiom的学习

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear,Sequential,CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import SGD

# 测试的数据集
dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64,shuffle=True)

# 面向于CIFAR10的神经网络训练函数（3，32x32）->（64，1x10）
class Sguan(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024,out_features=64),
            Linear(in_features=64,out_features=10)
        )

    def forward(self,input_py):
        output = self.module1(input_py)
        return output

loss = CrossEntropyLoss()
sguan_py = Sguan()
sguan_optim = torch.optim.SGD(sguan_loss.parameters(),lr=0.01,)
for epoch in range(20):
    result_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        # 1.卷积和池化
        img_out = sguan_py(imgs)
        result = loss(img_out,targets)

        # 2.优化器需要把梯度定义为零
        sguan_optim.zero_grad()

        # 3.输出结果需要一个反向传播,用到了backward内置函数
        result.backward()

        # 4.使用优化器优化卷积核参数
        sguan_optim.step()
        result_loss += result
    print(result_loss)





