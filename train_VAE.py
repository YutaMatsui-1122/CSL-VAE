from matplotlib import pyplot as plt
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch 
from dataset import Dataset_3dshapes
from torch.autograd import Variable
import numpy as np
import cv2
from betaVAE import betaVAE
import time
from utils import HSV2RGB, create_dataloader, VAE_model, save_model
import sys
import glob
from scipy.stats import kurtosis

batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01

########## create dataloader  (Check "shuffle = True". If this value is False, the model cannot gain disentangled representation) #############
dataset_size,_,_,dataloader= create_dataloader(dataset_name,batch_size,file_name_option,shuffle=True,dup_num=dup_num,shift_rate=shift_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta_list=[15,20]
latent_dim=10
linear_dim=1024
epoch=3000
prior_logvar_value = 0
loss_mode= "beta"  #"beta" or "beta-TC"
for iter in range(5):        
    for beta in beta_list:
        prior_mu = torch.zeros((dataset_size,latent_dim),dtype=torch.float).to(device)
        prior_logvar = torch.full((dataset_size,latent_dim),prior_logvar_value,dtype=torch.float).to(device)

        mutual_lerning_flag = False
        # setting of mutual learning
        model = VAE_model(beta=beta,latent_dim=latent_dim,linear_dim=linear_dim,device=device,dataset_name=dataset_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for i in range(epoch):
            train_loss = 0
            s=time.time()
            for batch_idx, (data, _) in enumerate(dataloader):
                data = Variable(data).to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar, z_d = model(data)
                batch_prior_mu = prior_mu[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_prior_logvar = prior_logvar[batch_idx*batch_size:(batch_idx+1)*batch_size]
                loss = model.loss_function(recon_batch, data, mu, logvar,z_d,batch_prior_mu,batch_prior_logvar,loss_mode=loss_mode, prior_logvar_value = prior_logvar_value,mutual_learning_flag=mutual_lerning_flag)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
            if i == 0 or (i+1) % 50 == 0 or i == (epoch-1):
                print('====> Beta: {} Epoch: {} Average loss: {:.4f}  Learning time:{}s'.format(beta, i+1, train_loss / len(dataloader.dataset),round(time.time()-s)))
                if mutual_lerning_flag:
                    save_model(model,loss_mode,latent_dim,beta,linear_dim, prior_logvar_value, i,file_name_option + "_Mutual_Learning")
                else:
                    save_model(model,loss_mode,latent_dim,beta,linear_dim, prior_logvar_value, i,file_name_option + f"_{iter}")