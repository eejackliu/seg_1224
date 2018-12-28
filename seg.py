import torchvision.datasets as dset
import torch
import torchvision
import  torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import sampler
from torch.utils.data import DataLoader as dataloader
import random
from voc_seg import my_data
import torchvision.models as model

import torchvision.transforms.functional as TF
from PIL import Image as image
import matplotlib.pyplot as plt

# from  matplotlib import pyplot as plt
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                    ])
mask_transform=transforms.Compose([transforms.ToTensor()])
trainset=my_data((240,320),'data',transform=image_transform,target_transform=mask_transform)
testset=my_data((240,320),'data',transform=image_transform,target_transform=mask_transform)
loader=dataloader(trainset,batch_size=4,shuffle=True)
vgg=model.vgg16(pretrained=True)
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
class Fcn(nn.Module):
    def __init__(self):
        super().__init__()
        self.numclass=21
        self.conv=vgg.features
        self.conv1=nn.Conv2d(512,self.numclass,1)
        nn.init.xavier_uniform_(self.conv1.weight)
        # nn.init.xavier_normal_(self.conv1.weight)
        self.tran_conv=nn.ConvTranspose2d(self.numclass,self.numclass,64,32,16)
        self.tran_conv.weight=torch.nn.Parameter(self.bilinear_kernel(21,21,64))
    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size),
            dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        weight=torch.from_numpy(weight)
        weight.requires_grad=True
        return weight
    def forward(self, input):
        x=self.conv(input)
        # print("forward")
        # print (x.shape)
        x=self.conv1(x)
        # print("one")
        # print (x.shape)
        x=self.tran_conv(x)
        return x
def train_fcn():

    fcn=Fcn()
    # fcn.train()
    criterion=nn.CrossEntropyLoss()
    optimize=torch.optim.Adam(fcn.parameters(),lr=0.001)
    # fcn=fcn.to(device)
    for i in range(1):
        for img,tag in  loader:
            # print ("input shape")
            # print(img.shape)
            # label_test=lable_data.numpy().argmax(axis=1)
            # lable_data=torch.unsqueeze(lable_data,1)
            input_data,lable_data=img.to(device),tag.to(device)
            optimize.zero_grad()
            output=fcn(input_data)
            # print("output shape")
            # print(output.size())
            # print("label size")
            # print(lable_data.size())
            # loss_=f.softmax(output,dim=1)
            loss=criterion(output,lable_data)
            print (loss)
            loss.backward()
            optimize.step()
            # break
        break
    return  fcn

voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def test(model):
    tmp=0
    with torch.no_grad():
        model.to(device)
        img,tag=dataloader(testset,batch_size=4)
        output=model(img)
        label=output.argmax(dim=1)
        tmp=label.numpy()
    img=label2image(tmp)
    picture(img,de_normal=True)
    picture(tag)
def label2image(pred):

    colormap=np.array(voc_colormap)
    return colormap[pred]
def picture(img,de_normal=False):
    # torchvision.utils.make_grid() picture must be numpy
    mean,std=np.array((-0.485, -0.456, -0.406)),np.array((1/0.229, 1/0.224, 1/0.225))
    image=img.transpose(0,2,3,1)
    num=len(image)
    tmp=image
    if de_normal:
         mean,std=np.array(mean),np.array(std)
         mean,std=np.tile(mean,(num,1)),np.tile(std,(num,1))
         tmp=image*std+mean
    for i,j in enumerate(tmp,1):
        plt.subplot(num,i,1)
        plt.imshow(j)
    plt.show()


model=train_fcn()
test(model)















# tmp=next(iter(loader))
# # TF.to_pil_image(TF.normalize(tmp[0][0],(-0.485, -0.456, -0.406),(1/0.229, 1/0.224, 1/0.225))).convert('RGB').show()
# mean,std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
# aa=tmp[1][0].numpy().transpose(1,2,0)*std+mean
# plt.imshow(aa)
# plt.show()



# path='data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
# with open(path) as f:
#         a=f.readlines()
#         path_b='data/VOCdevkit/VOC2012/JPEGImages/'
#         h=w=500
#         for i in a:
#                 m,n=image.open(path_b+i.rstrip('\n')+'.jpg').size
#                 h=m if h>m else h
#                 w=n if w>n else n
# print h,w