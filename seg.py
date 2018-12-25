import torchvision
a=torchvision.datasets.VOCSegmentation('data',download=True)
b=torchvision.datasets.VOCSegmentation('data',image_set='test',download=True)