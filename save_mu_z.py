from matplotlib import markers, pyplot as plt
from pyparsing import col
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch 
from scipy.stats import norm
from dataset import Dataset_3dshapes
from torch.autograd import Variable
import numpy as np
import cv2
from betaVAE import betaVAE
import time
from tqdm import tqdm
from utils import create_dataloader,VAE_model,HSV2RGB

batch_size=500
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01

########## create dataloader  (Check "shuffle = True". If this value is False, the model cannot gain disentangled representation) #############
dataset_size,_,_,dataloader= create_dataloader(dataset_name,batch_size,file_name_option,shuffle=False,dup_num=dup_num,shift_rate=shift_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mutual_learning_flag = False
if mutual_learning_flag:
    file_name_option+="_Mutual_Learning"

beta_list=[20]
latent_dim_list=[10]
linear_dim=1024
start_epoch=0
epoch=3000
prior_logvar = 2
loss_mode_list = ["beta"]  #"beta" or "beta-TC"
for iter in range(1):
    file_name_option_iter = file_name_option + f"_{iter}"
    for loss_mode in loss_mode_list:
        for beta in beta_list:
            for latent_dim in latent_dim_list:
                model=VAE_model(beta=beta,latent_dim=latent_dim,linear_dim=linear_dim,device=device,dataset_name=dataset_name)
                model.load_state_dict(torch.load(f"model_{loss_mode}VAE/main/latent_dim={latent_dim}/beta={beta}_linear={linear_dim}_prior_logvar={prior_logvar}_epoch={epoch}_{file_name_option_iter}.pth"))
                model.eval()
                mu_hist=np.empty((0,latent_dim))
                logvar_hist=np.empty((0,latent_dim))
                z_hist=np.empty((0,latent_dim))
                label_hist=np.empty((0,6))
                for num_batch, (data,label) in enumerate(dataloader):        
                    recon_x,mu,logvar,z = model(data.to(device))
                    recon_x,mu,logvar,z = recon_x.to("cpu").detach().numpy(), mu.to("cpu").detach().numpy(), logvar.to("cpu").detach().numpy(), z.to("cpu").detach().numpy()
                    mu_hist = np.append(mu_hist,mu,axis=0)
                    logvar_hist = np.append(logvar_hist,logvar,axis=0)
                    z_hist = np.append(z_hist,z,axis=0)
                    label_hist = np.append(label_hist,label.numpy(),axis=0)
                
                np.save(f"save_z/{loss_mode}_z_mu_beta={beta}_latent_dim={latent_dim}_linear_dim={linear_dim}_prior_logvar={prior_logvar}_{dataset_name}_{file_name_option_iter}",{"z":z_hist,"mu":mu_hist},allow_pickle=True)
        np.save(f"save_z/label_{dataset_name}_{file_name_option_iter}",label_hist)