from matplotlib import pyplot as plt
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch 
from dataset import Dataset_3dshapes
from torch.autograd import Variable
import numpy as np
import glob
import cv2
from betaVAE import betaVAE
from utils import HSV2RGB,VAE_model,create_dataloader
import matplotlib.animation as animation
from VAE_Module import VAE_Module

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

batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01

########## create dataloader  (Check "shuffle = True". If this value is False, the model cannot gain disentangled representation) #############
D,_,_,dataloader,shuffle_dataloader= create_dataloader(dataset_name,batch_size,file_name_option,dup_num=dup_num,shift_rate=shift_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta=16
latent_dim=10
linear_dim=1024
start_epoch=0
epoch=2999
prior_logvar = 0
mutual_iteration = 3
image_size = 64

for iter in range(mutual_iteration):
    model = VAE_Module(beta,epoch,latent_dim,linear_dim,image_size,batch_size,D,dataloader,shuffle_dataloader).to(device)
    model_parameter_file= glob.glob(f"model_VAE_Module/beta={beta}_mutual_iteration={iter}_epoch=101.pth")[0]
    print(model_parameter_file)
    model.load_state_dict(torch.load(model_parameter_file))
    model.eval()
    z_trans=np.arange(100) #Latent traversal range
    I=10
    fig, axes = plt.subplots(I,latent_dim+2, figsize=((16,9)))
    z_hist=np.empty((0,latent_dim))
    mu_hist=np.empty((0,latent_dim))
    for num_batch, (data,label) in enumerate(shuffle_dataloader):
        recon_x,mu,logvar,z = model(data.to(device))
        recon_x,mu,logvar,z = recon_x.to("cpu").detach().numpy(), mu.to("cpu").detach().numpy(), logvar.to("cpu").detach().numpy(), z.to("cpu").detach().numpy()
        z_hist = np.append(z_hist,z,axis=0)
        mu_hist = np.append(mu_hist,mu,axis=0)                
    traversal_z_list = np.array([np.linspace(np.min(mu_hist[:,i]),np.max(mu_hist[:,i]),len(z_trans)) for i in range(latent_dim)])
    fig2, axes2 = plt.subplots(2,5, figsize=((16,9)))
    for a in range(10):
        bins = np.linspace(np.min(z_hist[:,a]),np.max(z_hist[:,a]),100)
        axes2[a//5][a%5].hist(z_hist[:,a],bins=bins)
    fig2.savefig(f"result_VAE_Module/hist_beta={beta}_mutual_iteration={iter}.png")
    ani = animation.FuncAnimation(fig, plot_anime,frames=len(z_trans)-1, interval=50)
    save_file_name=f"result_VAE_Module/beta={beta}_mutual_iteration={iter}.gif"
    ani.save(save_file_name)
    print(f"save:{save_file_name}")