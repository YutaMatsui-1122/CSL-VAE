from matplotlib import pyplot as plt
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch 
from dataset import Dataset_3dshapes
from torch.autograd import Variable
import numpy as np
import glob
import cv2,os
from betaVAE import betaVAE
from utils import HSV2RGB,VAE_model,create_dataloader
import matplotlib.animation as animation
from VAE_Module import VAE_Module
from CSL_VAE import CSL_VAE


debug = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_anime(n):
    t=z_trans[n]       
    plt.suptitle(r"$z_i=$"+f"{t.round(1)}".rjust(4),fontsize=20)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if i ==0 and j!=0:
                #axes[i][j].set_ylabel(f'z{i+1}')
                axes[i][j].set_title(r'$z_{'+'{}'.format(j-1)+r'}$',fontsize=10)
    axes[0][0].set_title('original',fontsize=10)
    axes[0][1].set_title('recon',fontsize=10)

    # まずは入力画像を描画
    for i, im in enumerate(data[:axes.shape[0]]):
        im = HSV2RGB(im)
        axes[i][0].imshow(im)
        axes[i][0].set_ylabel(f'data{i+1}',fontsize=10)

    # 次に純粋な再構成画像
    _,mu,var,z = model(data.to(device))
    recon_y=model.decode(z)

    for i, im in enumerate(recon_y): #行のfor文
        if i == I:
            break
        im = HSV2RGB(im)
        axes[i][1].imshow(im)

    # Latent traversal    
    for j in range(latent_dim): #列のfor文
        _,mu,var,z = model(data.to(device))
        z[:,j]=traversal_z_list[j][t]
        y=model.decode(z)
        for i, im in enumerate(y): #行のfor文
            if i == I:
                break
            im = HSV2RGB(im)
            axes[i][j+2].imshow(im)

def latent_traversal_fig(data,model,data_min_list, data_max_list):
    traversal_num = 10
    latent_dim = 10
    _,mu,var,z = model(data.to(device))
    
    for d in range(latent_dim):
        fig, axes = plt.subplots(1,traversal_num, figsize=((16,9)))
        traversal_z_list = np.linspace(data_min_list[d],data_max_list[d],traversal_num)
        for t in range(traversal_num):
            z_traversal = z.to("cpu").detach()
            z_traversal[0][d] = traversal_z_list[t]
            recon_y=model.decode(z_traversal)
            axes[t].imshow(HSV2RGB(recon_y[0].numpy()))
        plt.savefig(result_MI_dir,f"z_{d}.pdf")
    

batch_size = 3000
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "six_view_point_66644_2"

########## create dataloader  (Check "shuffle = True". If this value is False, the model cannot gain disentangled representation) #############
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp = 8
exp_dir = f"exp_CSL_VAE/exp{exp}"
model_dir = os.path.join(exp_dir,"model")
result_dir = os.path.join(exp_dir,"result")
os.makedirs(result_dir,exist_ok=True)
word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
data = np.load(word_sequence_setting_file,allow_pickle=True).item()
valid_list,truth_F,truth_T0,truth_T = data.values()
dataloader,valid_loader,shuffle_dataloader,w,_,_,_,_,_= create_dataloader(batch_size,file_name_option,valid_list)
latent_dim = 10
w = w.numpy()
for iter in [0]:
    result_MI_dir = os.path.join(result_dir,f"MI{iter}")
    os.makedirs(result_MI_dir,exist_ok=True)
    CSL_Module_parameters,VAE_Module_parameters = np.load(f"exp_CSL_VAE/exp{exp}/Hyperparameters.npy",allow_pickle=True).item().values()
    csl_vae = CSL_VAE(file_name_option,mutual_iteration_number=11,CSL_Module_parameters=CSL_Module_parameters,VAE_Module_parameters=VAE_Module_parameters,valid_list=valid_list)
    csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir)
    model = csl_vae.vae_module.to(device)
    z_trans=np.arange(100) #Latent traversal range
    I=5
    fig, axes = plt.subplots(I,latent_dim+2, figsize=((16,9)))
    z_hist=np.empty((0,latent_dim))
    mu_hist=np.empty((0,latent_dim))
    for num_batch, (data,label,_) in enumerate(dataloader):
        recon_x,mu,logvar,z = model(data.to(device))
        recon_x,mu,logvar,z = recon_x.to("cpu").detach().numpy(), mu.to("cpu").detach().numpy(), logvar.to("cpu").detach().numpy(), z.to("cpu").detach().numpy()
        z_hist = np.append(z_hist,z,axis=0)
        mu_hist = np.append(mu_hist,mu,axis=0)
    data = data[[300,500,1000,1333,2666]]
    traversal_z_list = np.array([np.linspace(np.min(mu_hist[:,i]),np.max(mu_hist[:,i]),len(z_trans)) for i in range(latent_dim)])
    ani = animation.FuncAnimation(fig, plot_anime,frames=len(z_trans)-1, interval=50)
    save_file_name=os.path.join(result_MI_dir,"latent_traversal.gif")
    ani.save(save_file_name)
    print(f"save:{save_file_name}")