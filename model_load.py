'''
Author: 星必尘Sguan
Date: 2025-04-25 19:57:32
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-25 20:59:43
FilePath: \test_TrainPytorch\model_load.py
Description: 网络模型的保存与读取(重在读取)

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
import torchvision
import torch

# 加载方式一
model1 = torch.load("vgg16_method1.pth",weights_only=False)
print(model1)

# 加载方式二（官方推荐）
model2 = torch.load("vgg16_method2.pth",weights_only=False)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(model2)
print(vgg16)


