from PIL import Image as image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import torchvision.datasets as dset
class my_data(torch.utils.data.Dataset):
    def __init__(self,data_size,root,image_set='train',transform=None,target_transform=None):
        self.shape=data_size
        self.root=os.path.expanduser(root)
        self.transform=transform
        self.target_transform=target_transform
        voc_dir=os.path.join(self.root,'VOCdevkit/VOC2012')
        image_dir=os.path.join(voc_dir,'JPEGImages')
        mask_dir=os.path.join(voc_dir,'SegmentationClass')
        splits_dir=os.path.join(voc_dir,'ImageSets/Segmentation')
        splits_f=os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(splits_f),'r') as f:
            file_name=[x.strip() for x in f.readlines()]
        self.image=[os.path.join(image_dir,x+'.jpg') for x in file_name]
        self.mask=[os.path.join(mask_dir,x+'.png') for x in file_name]
        assert (len(self.image)==len(self.mask))

    def __getitem__(self, index):
        img=image.open(self.image[index]).convert('RGB')
        target=image.open(self.mask[index])
        i,j,h,w=transforms.RandomCrop.get_params(img,self.shape)
        img=TF.crop(img,i,j,h,w)
        target=TF.crop(target,i,j,h,w)
        return self.transform(img),self.target_transform(target)
    def __len__(self):
        return len(self.image)