'''
Author: 星必尘Sguan
Date: 2025-04-24 16:13:07
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 16:43:25
FilePath: \test_TrainPytorch\Dataset-Transfroms.py
Description: dataset和transfroms联合使用(用到了torchvision里面的数据集)

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

trans_dataset = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

trans_set = torchvision.datasets.CIFAR10(root=".\\dataset",train=True,transform=trans_dataset,download=True)
test_set = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,transform=trans_dataset,download=True)

# print(test_set[0])
writer = SummaryWriter("logs")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()
