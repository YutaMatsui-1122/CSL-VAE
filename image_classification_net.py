import torch,os,glob,time,math,cv2
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import save_model,HSV2RGB,create_dataloader
from utils import label_to_word
from torchinfo import summary


class Image_Classification_Net(nn.Module):
    def __init__(self,V):
        super(Image_Classification_Net, self).__init__()
        layer = 4
        image_channel=3
        image_size = 64
        channels=[image_channel,32,64,128,256]
        kernel = np.repeat(4,layer)
        stride = np.repeat(2,layer)
        padding = np.repeat(1,layer)
        modules=[]
        for h in range(len(channels)-1):
            modules.append(
                nn.Conv2d(in_channels=channels[h], out_channels=channels[h+1], kernel_size=kernel[h], stride=stride[h], padding=padding[h])
            )
            modules.append(
                nn.ReLU()
            )
        Linear1dim = int((image_size/(2**layer))*(image_size/(2**layer))*channels[-1])
        Linear2dim = Linear1dim//2
        Linear3dim = Linear2dim//2
        Linear4dim = Linear3dim//2
        modules.append(nn.Linear(Linear1dim, Linear2dim))
        modules.append(nn.Linear(Linear2dim, Linear3dim))
        modules.append(nn.Linear(Linear3dim, Linear4dim))
        modules.append(nn.Linear(Linear4dim, V))
        modules.append(nn.Sigmoid())
        self.net = nn.Sequential(*modules)
    def forward(self,input):
        self.net(input)

batch_size = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "five_view_point_55544"
word = label_to_word(file_name_option)
dup_num = 2
shift_rate = 0.01
file_name_option += f"×{dup_num}_shift_{shift_rate}"
exp = 25
exp_dir = f"exp_CSL_VAE/exp{exp}"
word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
data = np.load(word_sequence_setting_file,allow_pickle=True).item()
valid_list,grammar_list,Nd_rate,truth_F,truth_T0,truth_T = data.values()
dataloader,valid_loader,shuffle_dataloader,truth_category,full_dataloader,full_shuffle_loader = create_dataloader(batch_size,file_name_option,valid_list)

for image,label,_ in full_shuffle_loader:
    pass

icn = Image_Classification_Net(V=23)
summary(
    icn,
    input_size=(batch_size, 3, 224, 224),
    col_names=["output_size", "num_params"],
)