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
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "three_view_point_88844"
word = label_to_word(file_name_option)
'''dup_num = 10
shift_rate = 0.01
file_name_option += f"×{dup_num}_shift_{shift_rate}"
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta=20
latent_dim=10
linear_dim=1024
start_epoch=0
epoch=3000
prior_logvar = 2
loss_mode_list = ["beta"]  #"beta" or "beta-TC"
mutual_iteration = 6

def calc_ARI(label,truth_labels):
  ARI_list = []
  ARI_index = []
  for a in range(truth_labels.shape[1]):
    ARI_list.append(np.max([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
    ARI_index.append(np.argmax([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
  return np.round(ARI_list,decimals=3), ARI_index
def logpdf(z,mu,lam):
    return -0.5 * (lam*(z-mu)**2-np.log(lam)+np.log(2*math.pi))
for iter in range(mutual_iteration):
    exp = 8
    iter = 10
    valid_list = [[3,6,11,16,20],[0,5,14,17,22],[4,7,12,16,19],[1,9,10,18,21]]
    exp_dir = f"exp_CSL_VAE/exp{exp}"
    model_dir = os.path.join(exp_dir,"model")
    result_dir = os.path.join(exp_dir,"result")
    os.makedirs(result_dir,exist_ok=True)
    word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
    data = np.load(word_sequence_setting_file,allow_pickle=True).item()
    valid_list,grammar_list,Nd_rate = data.values()
    dataloader,valid_loader,shuffle_dataloader,truth_category= create_dataloader(batch_size,file_name_option,valid_list)
    existing_file = glob.glob(os.path.join(model_dir,f"CSL_Module/iter=*_mutual_iteration={iter}.npy"))[0]
    data = np.load(existing_file,allow_pickle=True).item()
    truth_label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
    F = data["F"]
    z = data["z"]
    w = data["w"]
    lam = data["lam"]
    mu =data["mu"]
    theta = data["theta"]
    T = data["T"]
    T0 = data["T0"]
    pi = data["pi"]
    c = data["c"]
    #plt.figure()
    if len(theta.shape)==2:
        heat_theta = theta
    else:
        heat_theta = np.reshape(theta,(-1,theta.shape[2]))
    sns.heatmap(heat_theta)
    print(T0)
    plt.figure()
    sns.heatmap(T)
    plt.show()
    print(truth_category.shape)
    print("ARI a:",calc_ARI(c,truth_category))
    truth_F = np.tile(np.arange(5),(F.shape[0],1))
    print("ARI F:",calc_ARI(F,truth_F))
    cw= np.stack((c[:,8][:20000],w[:,1][:20000])).T
    np.set_printoptions(threshold=np.inf)
    '''for i in range(20):
        print(cw[i*1000:(i+1)*1000])
        print(i)
        plt.figure()
        #plt.show()'''
    plt.figure()
    for a in range(10):            
        bins = np.linspace(np.min(z[:,a]),np.max(z[:,a]),100)
        for k in range(15):
            index = np.where(c[:,a]==k)[0]
            z_a_k = z[:,a][index]
            plt.hist(z_a_k,bins = bins,alpha=0.5,label=f"c={k+1}")
        plt.xlabel(r"$z$")
        plt.legend()
        plt.show()