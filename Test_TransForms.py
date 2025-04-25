'''
Author: 星必尘Sguan
Date: 2025-04-24 12:23:51
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 13:32:12
FilePath: \test_TrainPytorch\Test_TransForms.py
Description: Transforms的使用

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "dataset\\train\\ants\\0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
 

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)
writer.add_image("Tensor_img",tensor_img)
writer.close()

