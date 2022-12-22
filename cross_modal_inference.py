from CSL_VAE import CSL_VAE
from VAE_Module import VAE_Module
from CSL_Module_ak import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob
from utils import *
import numpy as np
import matplotlib.pyplot as plt


batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01

Attribute_num = 5

word = label_to_word(file_name_option)
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

for data,label,_ in csl_vae.vae_module.shuffle_dataloader:
    break

label = label.numpy()[:,:5]
o_star = data[0]
N_star = 5
mutual_iteration = 0
print(np.round(csl_vae.csl_module.T,decimals=3))
print(np.round(csl_vae.csl_module.T0,decimals=3))
for i in range(100):
    w_star = label[i]
    #w_star = csl_vae.img2wrd(o_star,N_star,mutual_iteration)
    o_star = csl_vae.wrd2img(w_star,mutual_iteration)
    fig=plt.figure(figsize=((8,4)))
    word_sequence = ",  ".join(word[label[i]])
    print(word_sequence)
    fig.suptitle("wrd2img inference")
    axes1 = fig.add_subplot(1,2,1)
    axes2 = fig.add_subplot(1,2,2)
    axes1.tick_params(bottom=False, left=False, right=False, top=False);axes2.tick_params(bottom=False, left=False, right=False, top=False)
    axes1.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False);axes2.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

    axes1.imshow(HSV2RGB(data[i]))
    axes1.set_title("correct image")
    axes2.imshow(HSV2RGB(o_star))
    axes2.set_title("infered image by CSL+VAE")
    plt.show()

    figure = plt.figure(figsize=(4,4))
    w_star = csl_vae.img2wrd(o_star,N_star,mutual_iteration)

    word_sequence = ",  ".join(word[w_star])
    print("infered word sequence \"",word_sequence,"\"")
    plt.title("img2wrd inference")
    plt.imshow(HSV2RGB(data[i]))
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.show()

print(csl_vae.csl_module.theta)