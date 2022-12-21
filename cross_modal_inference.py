from CSL_VAE import CSL_VAE
from VAE_Module import VAE_Module
from CSL_Module import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob
from utils import create_dataloader,label_to_vocab,HSV2RGB
import numpy as np
import matplotlib.pyplot as plt

batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01

Attribute_num = 5

file_name_option += f"×{dup_num}_shift_{shift_rate}"
label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
w=label_to_vocab(label[:,:5])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 16
latent_dim=10
linear_dim=1024
epoch=3000
image_size = 64

csl_vae = CSL_VAE(w,beta,file_name_option)
#csl_vae.learn()
csl_vae.setting_learned_model(w,beta=16,mutual_iteration=2)

for data,label,_ in csl_vae.vae_module.dataloader:
    break

label = label.numpy()[:,:5]
N_star = 5
mutual_iteration = 2
w_star = [0,5,10,15,19]
for i in range(100):
    w_star = csl_vae.wrd2img(w_star,mutual_iteration)
    print(w_star,label[i])

print(csl_vae.csl_module.theta)