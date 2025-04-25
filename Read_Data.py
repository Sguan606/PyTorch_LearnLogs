'''
Author: 星必尘Sguan
Date: 2025-04-24 10:33:43
LastEditors: 星必尘Sguan|3464647102@qq.com
LastEditTime: 2025-04-24 11:28:50
FilePath: \test_TrainPytorch\Read_Data.py
Description: 学习DaTaset类代码实战

Copyright (c) 2025 by $JUST, All Rights Reserved. 
'''
from torch.utils.data import Dataset
# import cv2
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir =label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset\\train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
train_dataset = ants_dataset + bees_dataset

img, label = train_dataset[5]
img.show()
    
