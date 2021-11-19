import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.set_figsize()
img=d2l.Image.open('./img/cat.jpg')
d2l.plt.imshow(img)

def apply(img,aug,num_rows=2,num_cols=4,scale=2.5):
    Y=[aug(img) for _ in range(num_rows*num_cols)]
    d2l.show_images(Y,num_rows,num_cols,scale=scale)


# apply(img,torchvision.transforms.RandomHorizontalFlip())

# apply(img,torchvision.transforms.RandomVerticalFlip())

shape_aug=torchvision.transforms.RandomResizedCrop((200,200),scale=(0.1,1),ratio=(0.5,2))   #(200,200) risize后的尺寸大小|scale裁剪的窗口尺寸占总尺寸的多少|rario，高宽比，一般维1：2与2：1
# apply(img,shape_aug)

#brightness亮度增加或减少50% contrast对比度 saturation saturation饱和度 hue色调
# apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0,saturation=0,hue=0))
# apply(img,torchvision.transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0.5))
# apply(img,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5))

augs=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),shape_aug,torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)])
apply(img,augs)


plt.show()