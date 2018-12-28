from PIL import Image as image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import torchvision.datasets as dset
voc_colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
class my_data(torch.utils.data.Dataset):
    def __init__(self,data_size,root,image_set='train',transform=None,target_transform=False):
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

        self.class_index=np.zeros(256**3)
        for i,j in enumerate(voc_colormap):
            tmp=(j[0]*256+j[1])*256+j[2]
            self.class_index[tmp]=i

    def __getitem__(self, index):
        img=image.open(self.image[index]).convert('RGB')
        target=image.open(self.mask[index])
        i,j,h,w=transforms.RandomCrop.get_params(img,self.shape)
        img=TF.crop(img,i,j,h,w)
        target=TF.crop(target,i,j,h,w)
        if self.target_transform :
            return self.transform(img),target
        target=np.array(target)
        target=(target[0]*256+target[1])*256+target[2]
        target=self.class_index[target]
        return self.transform(img),target

    def __len__(self):
        return len(self.image)