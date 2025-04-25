'''
Author: 星必尘Sguan
Date: 2025-04-24 16:49:17
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 19:41:07
FilePath: \test_TrainPytorch\Dataloader.py
Description: 学习Dataloader的使用

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision


# 准备的一个测试集
test_data = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)
# batch_size是指每次取数据多少个
# drop_last最后几张不足batch_size的时候，是否舍去
# shuffle是否要打乱每次取数据的顺序

# 测试数据集中第一张图片及targe
img,targe = test_data[0]
print(img.shape)
print(targe)

writer = SummaryWriter("logs")

#DataLoader中取样本数据
step = 0
for data in test_loader:
    imgs,targes = data
    # print(imgs.shape)
    # print(targes)
    writer.add_images("test_data",imgs,step)
    step += 1

writer.close()