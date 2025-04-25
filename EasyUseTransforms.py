'''
Author: 星必尘Sguan
Date: 2025-04-24 13:36:59
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 19:55:14
FilePath: \test_TrainPytorch\EasyUseTransforms.py
Description: 常用的一些Transforms函数

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset\\train\\ants\\6743948_2b8c096dda.jpg")

# ToTensor类型转换
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

# Normalize归一化
trans_norm = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]) # 提供3个标准差
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize",img_norm)

# Resize强行修改图片尺寸
trans_resize = transforms.Resize((512,512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize)

# Compose-resize-2 仅对图片高缩减到512,使用的是等比压缩
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

# RandomCrop随机裁剪一个指定尺寸大小的图片
trans_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_random = trans_compose_2(img)
    writer.add_image("RandomCrop",img_random,i)


writer.close()
