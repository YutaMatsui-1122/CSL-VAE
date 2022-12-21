import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import adjusted_rand_score as ARI
import torch 
from dataset import Dataset_3dshapes
from torch.autograd import Variable
import cv2
from betaVAE import betaVAE
from utils import HSV2RGB,VAE_model,create_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "one_view_point_55544"
dup_num = 10
shift_rate = 0.01
file_name_option += f"×{dup_num}_shift_{shift_rate}"
D,_,_,dataloader,shuffle_dataloader= create_dataloader(batch_size,file_name_option)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta=20
latent_dim=10
linear_dim=1024
start_epoch=0
epoch=3000
prior_logvar = 2
loss_mode_list = ["beta"]  #"beta" or "beta-TC"
mutual_iteration = 3

def calc_ARI(label,truth_labels):
    ARI_list = [ARI(label,truth_labels[:,i]) for i in range(truth_labels.shape[1])]
    return np.argmax(ARI_list),np.max(ARI_list)

for num_batch, (data,label,_) in enumerate(dataloader):
    break
for iter in range(mutual_iteration):
    iter = 2
    existing_file = glob.glob(f"save_CSL+VAE/iter=*_mutual_iteration={iter}.npy")[0]
    data = np.load(existing_file,allow_pickle=True).item()
    print(existing_file)
    truth_label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
    F = data["F"]
    z = data["z"]
    lam = data["lam"]
    mu =data["mu"]
    theta = data["theta"]
    c = data["c"]
    print(lam)
    for m in mu:
        print(m)
    print(F)
    plt.figure()
    sns.heatmap(theta)
    plt.show()
    plt.figure()
    for a in range(10):            
        bins = np.linspace(np.min(z[:,a]),np.max(z[:,a]),100)
        for k in range(7):
            index = np.where(c[:,a]==k)[0]
            z_a_k = z[:,a][index]
            plt.hist(z_a_k,bins = bins,alpha=0.5)            
        print("ARI a:",calc_ARI(c[:,a],truth_label))
        plt.show()
    mu_hat = np.array([mu[a][c[:,a]] for a in range(10)]).T
    Sigma_hat = np.array([1/lam[a][c[:,a]] for a in range(10)]).T
    print("mu",mu_hat[:30])
    print(Sigma_hat[:30])
    phi_hat = [mu_hat,Sigma_hat]
    print(phi_hat)
