from VAE_Module import VAE_Module
from CSL_Module_ak import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob
from utils import create_dataloader,label_to_vocab,HSV2RGB
import numpy as np
import matplotlib.pyplot as plt

class CSL_VAE():
    def __init__(self,w,beta,file_name_option) -> None:
        self.mutual_iteration_number = 3
        self.phi = []
        self.w=w
        self.beta = beta
        self.epoch = 5000
        self.latent_dim = 10
        self.linear_dim = 1024
        self.image_size = 64
        self.batch_size = 500
        self.file_name_option = file_name_option
        self.D,_,_,dataloader,shuffle_dataloader= create_dataloader(self.batch_size,self.file_name_option)
        self.vae_module = VAE_Module(self.beta,self.epoch,self.latent_dim,self.linear_dim,self.image_size,self.batch_size,self.D,dataloader,shuffle_dataloader)
        self.csl_module = CSL_Module()
    
    def setting_learned_model(self,w,beta,mutual_iteration):
        model_file = glob.glob(f"model_VAE_Module/beta={beta}_mutual_iteration={mutual_iteration}_epoch=*.pth")[0]
        self.vae_module.load_state_dict(torch.load(model_file))
        self.csl_module.setting_parameters(w,z=np.zeros((self.D,self.latent_dim)),mutual_iteration=mutual_iteration)
    
    def learn(self):
        for i in range(self.mutual_iteration_number):
            z=self.vae_module.learn(i)
            phi_hat = self.csl_module.learn(self.w,z,i)
            self.vae_module.prior_mu=torch.from_numpy(phi_hat[0]).clone().to(device)            
            self.vae_module.prior_logvar=torch.from_numpy(phi_hat[1]).clone().to(device).log()
        
    def img2wrd(self,o_star,N_star,mutual_iteration):
        self.setting_learned_model(self.w,self.beta,mutual_iteration)
        o_star = torch.unsqueeze(o_star,0)
        z_star,_,_ = self.vae_module.encode(o_star)
        z_star = z_star.detach().numpy()[0]
        c_star = self.csl_module.img2wrd_sampling_c(z_star)
        F_star = self.csl_module.img2wrd_sampling_F(N_star)
        w_star = self.csl_module.img2wrd_sampling_w(c_star,F_star,N_star)
        
        return w_star

    def wrd2img(self,w_star,mutual_iteration):
        self.setting_learned_model(self.w,self.beta,mutual_iteration)        
        F_star = self.csl_module.wrd2img_sampling_F(w_star)
        c_star = self.csl_module.wrd2img_sampling_c(w_star, F_star)
        z_star = self.csl_module.wrd2img_sampling_z(c_star)
        z_star = torch.from_numpy(z_star).float()
        z_star = torch.unsqueeze(z_star,0)
        o_star = self.vae_module.decode(z_star)[0]
        return o_star

if __name__ == "__main__":
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
    csl_vae.learn()