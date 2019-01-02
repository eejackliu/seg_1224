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
trainset=my_data((96,224),'data',transform=image_transform)
testset=my_data((96,224),'data',transform=image_transform,target_transform=mask_transform)
loader=dataloader(trainset,batch_size=4,shuffle=True)
test_loader=dataloader(testset,batch_size=4,)
vgg=model.vgg16(pretrained=True)
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device=torch.device('cpu')
dtype = torch.float32

class Fcn(nn.Module):
    def __init__(self):
        super(Fcn,self).__init__()
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
    fcn=fcn.to(device)
    for i in range(1):
        for img,tag in  loader:
            # print ("input shape")
            # print(img.shape)
            # label_test=lable_data.numpy().argmax(axis=1)
            # lable_data=torch.unsqueeze(lable_data,1)
            input_data,lable_data=img.to(device,dtype=dtype),tag.to(device,dtype=torch.long)
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
            break
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
        img,tag=next(iter(test_loader))
        img=img.to(device)
        output=model(img)
        label=output.argmax(dim=1)
        tmp=label.cpu()
    img=label2image(tmp)

    pic_pred(img,tag)
def label2image(pred):

    colormap=np.array(voc_colormap)
    return colormap[pred]
def pic_pred(img,tag):
    num=len(img)
    tag=tag.numpy().transpose(0,2,3,1)
    tmp=np.concatenate((img,tag),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(2,num,i)
        plt.imshow(j)
    plt.show()
def picture(img,tag,de_normal=False):
    # torchvision.utils.make_grid() picture must be numpy
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    num=len(img)
    # N=len(img)
    tmp=img.transpose(0,2,3,1)
    tag=tag.numpy().transpose(0,2,3,1)
    if de_normal:
         tmp = img.transpose(0,2,3,1)
         # mean,std=np.tile(mean,(num,1)),np.tile(std,(num,1))
         tmp=tmp*std+mean
    tmp=np.concatenate((tmp,tag),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(2,num,i)
        plt.imshow(j)
    # _,ax=plt.subplots(1,num)
    # for i,j in enumerate(ax):
    #     j.imshow(tmp[i])
    plt.show()
#
# from skimage import io
# path='data/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'
# pict=image.open(path).convert('RGB')
# pic=io.imread(path)
# pic=np.array(pict)
# d=pic[150:200,0:50]
# f=d[40:42 ,40:42]
# model=train_fcn()
# torch.save(model.state_dict(),'model')
test_model=Fcn()
test_model.load_state_dict(torch.load('model'))
test(test_model)
#
# def bilinear_kernel(in_channels, out_channels, kernel_size):
#         factor = (kernel_size + 1) // 2
#         if kernel_size % 2 == 1:
#             center = factor - 1
#         else:
#             center = factor - 0.5
#         og = np.ogrid[:kernel_size, :kernel_size]
#         filt = (1 - abs(og[0] - center) / factor) * \
#                (1 - abs(og[1] - center) / factor)
#         weight = np.zeros(
#             (in_channels, out_channels, kernel_size, kernel_size),
#             dtype='float32')
#         weight[range(in_channels), range(out_channels), :, :] = filt
#         weight=torch.from_numpy(weight)
#         weight.requires_grad=True
#         return weight


# a=torch.zeros((1,3,224,320))
#
# b=vgg.features(a)
# print b.shape
# w1=torch.zeros((21,512,1,1))
# c=nn.functional.conv2d(b,w1)
# w2=bilinear_kernel(21,21,64)
# d=nn.functional.conv_transpose2d(c,w2,stride=32,padding=16)








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
#                 w=m if w>m else w
#                 h=n if h>n else h
# print h,w