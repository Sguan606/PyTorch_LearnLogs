'''
Author: 星必尘Sguan
Date: 2025-04-25 19:03:30
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 19:53:48
FilePath: \test_TrainPytorch\model_pretrained.py
Description: 现有网络模型VGG16的修改及使用

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torchvision
from torch.utils.data import Dataset
from torch import nn

vgg16_flase = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)


dataset = torchvision.datasets.CIFAR10(root=".\\dataset",train=False,
                                        transform=torchvision.transforms.ToTensor(),download=True)


vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


