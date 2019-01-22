import torchvision.datasets as dset
import torch
import torchvision
import  torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import sampler
from torch.utils.data import DataLoader as dataloader
import random
from voc_seg import my_data,voc_colormap
import torchvision.models as model
import torchvision.transforms.functional as TF
from PIL import Image as image
import matplotlib.pyplot as plt
import nonechucks as nc
# from  matplotlib import pyplot as plt
image_transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
                                    ])
mask_transform=transforms.Compose([transforms.ToTensor()])# to_tensor will make it from nhwc to nchw
trainset=my_data((500 ,500),'data',transform=image_transform)
trainset=nc.SafeDataset(trainset)
testset=my_data((96,224),'data',transform=image_transform)
# loader=dataloader(trainset,batch_size=32,shuffle=True)
loader=nc.SafeDataLoader(trainset,batch_size=32)
test_loader=dataloader(testset,batch_size=4)
vgg=model.vgg16(pretrained=True)
device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
# dtype = torch.float32
#%%
a,b=next(iter(loader))
class_num=21
#%%
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
        x=self.conv1(x)
        x=self.tran_conv(x)
        return x
def train_fcn():

    fcn=Fcn()
    fcn.train()
    # fcn.train()
    criterion=nn.CrossEntropyLoss()
    optimize=torch.optim.Adam(fcn.parameters(),lr=0.001)
    fcn=fcn.to(device)
    for i in range(156):
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
    return  fcn
def test(model):
    img_list=[]
    tag_list=[]
    seg_list=[]
    with torch.no_grad():
        model.to(device)
        model.eval()
        # img,tag=next(iter(test_loader))
        for img,tag in test_loader:
            img=img.to(device)
            img_list.append(img)
            tag_list.append(tag)
            output=model(img)
            label=output.argmax(dim=1)
            tmp=label.cpu()
            img_final=torch.from_numpy(label2image(tmp))
            seg_list.append(img_final)
        img=torch.cat(img_list,dim=0)
        tag=torch.cat(tag_list,dim=0)
        seg=torch.cat(seg_list,dim=0)
    score=iou(seg,tag)
    pic_pred(img[0:4].numpy(),tag[0:4].numpy(),seg[0:4].numpy())
def label2image(pred):
    colormap=np.array(voc_colormap)
    return colormap[pred]
def pic_pred(img,tag,pred):
    plt.figure()
    num=len(img)
    img=img.transpose(0,2,3,1) #because the broading-cast(numpy would increase the axes at the left,the anti-normal must at NHWC)
    # tag=tag.transpose(0,2,3,1)
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    img=img*std+mean
    tmp=np.concatenate((img,tag/255.0,pred/255.0),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    plt.show()
torchvision.utils.make_grid()
def my_iou(img,tag):
    img=img.long()
    tag=tag.long()
    intersaction=(img==tag)
    union=img+tag
    union=torch.where(union>0)
    iou=intersaction.float().sum()/union.float().sum()
    return iou/class_num
def iou(img,tag):
    img=img.to(device)
    tag=tag.to(device)
    img=img.long().view(-1)
    tag=tag.long().view(-1)

    cls_iou=[]
    for i in range(1,class_num):
        tmp_img=(img==i)
        tmp_tag=(tag==i)
        intersaction=tmp_tag[tmp_img].float().sum()
        union=tmp_img.float().sum()+tmp_tag.float().sum()-intersaction
        if intersaction ==0:
            cls_iou.append(torch.tensor(0.,device=device))
            continue
        cls_iou.append(intersaction.to(torch.float).sum()/union.sum())
    average=torch.tensor(cls_iou)
    return average.mean(),average
#model=train_fcn()
#torch.save(model.state_dict(),'model')
def picture(img,tag,de_normal=False):
    # torchvision.utils.make_grid() picture must be numpy
    plt.figure()
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    num=len(img)
    # N=len(img)
    # tmp=img.transpose(0,2,3,1)
    # tag=tag.transpose(0,2,3,1)
    if de_normal:
         tmp = img.transpose(0,2,3,1)
         # mean,std=np.tile(mean,(num,1)),np.tile(std,(num,1))
         tmp=tmp*std+mean
    tmp=np.concatenate((tmp,tag),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(2,num,i)
        plt.imshow(j)
    plt.show()

# test_model=Fcn()
# test_model.load_state_dict(torch.load('model',map_location='cpu'))
# # test(test_model)
# img_list=[]
# tag_list=[]
# seg_list = []
# model=test_model

# with torch.no_grad():
#     model.to(device)
#     model.eval()
#     # img,tag=next(iter(test_loader))
#     for img, tag in test_loader:
#         # img,tag=next(iter(test_loader))
#         img = img.to(device)
#         img_list.append(img)
#         tag_list.append(tag)
#         output = model(img)
#         label = output.argmax(dim=1)
#         tmp = label.cpu()
#         seg_list.append(tmp)
#
#     img = torch.cat(img_list, dim=0)
#     tag = torch.cat(tag_list, dim=0)
#     seg = torch.cat(seg_list, dim=0)
#     score,score_list = iou(seg, tag)
#     img_final = torch.from_numpy(label2image(tmp))
# pic_pred(img[0:4].cpu().numpy(), label2image(tag[0:4].numpy().astype(np.int)), label2image(seg[0:4].numpy()))

#picture(img[0:4].cpu().numpy(),label2image(tag[0:4].numpy().astype(np.int)),de_normal=True)

# #
# from skimage import io
# path='data/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'
# pict=image.open(path).convert('RGB')
# pic=io.imread(path)
# pic=np.array(pict)
# d=pic[150:200,0:50]
# f=d[40:42 ,40:42]
# model=train_fcn()
# torch.save(model.state_dict(),'model')


#

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