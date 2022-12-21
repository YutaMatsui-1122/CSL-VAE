from pickletools import int4
import cv2
from cv2 import COLOR_HSV2RGB
import numpy as np
from torchvision import transforms
import torch.utils.data
from matplotlib import pyplot as plt
import h5py,copy

def HSV2RGB(HSV_img):
    HSV_img=HSV_img.to("cpu").detach().numpy().transpose(1,2,0)
    RGB_img=copy.deepcopy(HSV_img)
    H_max = 255
    S_max = 255
    V_max = 255
    RGB_img[:,:,0]=HSV_img[:,:,0]*H_max
    RGB_img[:,:,1]=HSV_img[:,:,1]*S_max
    RGB_img[:,:,2]=HSV_img[:,:,2]*V_max
    RGB_img = RGB_img.astype(np.uint8)
    RGB_img = cv2.cvtColor(RGB_img,cv2.COLOR_HSV2BGR_FULL)
    return RGB_img

def label_to_vocab(label):
    attribute_num = label.shape[1]
    index_offset = 0
    num_W = np.sum([np.unique(label[:,i]).shape[0] for i in range(attribute_num)])
    for a in range(attribute_num):
        for i, category in enumerate(np.unique(label[:,a])):
            index = np.where(label[:,a]==category)[0]
            label[:,a][index] =i
        label[:,a] += index_offset
        index_offset += len(np.unique(label[:,a]))
    w = label.astype(np.int8)
    return w

class Dataset_3dshapes(torch.utils.data.Dataset):
    def __init__(self,data_filename,label_filename):
        self.image = np.load(data_filename).transpose((0,3,1,2))
        self.label = label_to_vocab(np.load(label_filename))
        self.image = torch.tensor(self.image,dtype=torch.float)
        self.label = torch.tensor(self.label,dtype=torch.int8)
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index1):
        data1=self.image[index1]
        label1=self.label[index1]
        return data1,label1,index1