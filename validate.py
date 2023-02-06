import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from heatmap import *
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp', required=True, help="experiment number")
parser.add_argument('--MI_list', required=True, nargs="*", type=int, help='a list of Mutual Inference Iteration number')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "six_view_point_66644_2"
word = label_to_word(file_name_option)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_ARI_EAR(label,truth_labels,F,truth_F):
    ARI_list = []
    ARI_index = []
    for a in range(truth_labels.shape[1]):        
        ARI_list.append(np.max([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
        ARI_index.append(np.argmax([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
    D,N_max = truth_F.shape
    A = np.max(F)
    N = np.array([np.count_nonzero(truth_F[d]<5) for d in range(D)])
    truth_F = np.array([[np.array(ARI_index)[truth_F[d][n]] if n<N[d] else A  for n in range(N_max)] for d in range(D)])
    EAR = np.sum([np.count_nonzero(truth_F[d][:N[d]]==F[d][:N[d]])for d in range(D)])/np.sum(N)
    return ARI_list, np.array(ARI_index),np.round(EAR,decimals=3)

def plot_category_image(result_dir,image,c,ARI_index,column_num=3, row_num=3):
    attribute_name_list = ["Floor color","Wall color","Object color","Size","Shape"]
    for attribute_name, a in zip(attribute_name_list,ARI_index):
        category_list = np.where(np.bincount(c[:,a],minlength=np.max(c)+1)!=0)[0]
        for k in category_list:
            index = np.where(c[:,a]==k)[0]
            images = image[index[np.random.randint(len(index),size = column_num*row_num)]]
            fig, axes = plt.subplots(row_num,column_num, figsize=((row_num*3,column_num*3)))
            #plt.suptitle(r"$(a=$"+f"{a}, "+r"$c=$"+f"{k})",size=20)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            i = 0
            for column in range(column_num):
                for row in range(row_num):
                    axes[column,row].imshow(HSV2RGB(images[i]))
                    axes[column,row].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                    axes[column,row].tick_params(bottom=False, left=False, right=False, top=False)
                    i += 1
            save_file_name = f"a={a},c={k}.svg"
            plt.savefig(os.path.join(result_dir,save_file_name))

def logpdf(z,mu,lam):
    return -0.5 * (lam*(z-mu)**2-np.log(lam)+np.log(2*math.pi))


exp = args.exp

for iter in args.MI_list:
    print("Mutual Iteration:",iter)
    
    exp_dir = f"exp_CSL_VAE/exp{exp}"
    model_dir = os.path.join(exp_dir,"model")
    result_dir = os.path.join(exp_dir,"result")
    result_MI_dir = os.path.join(result_dir,f"MI{iter}")
    os.makedirs(result_dir,exist_ok=True)
    os.makedirs(result_MI_dir,exist_ok=True)
    word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
    data = np.load(word_sequence_setting_file,allow_pickle=True).item()
    valid_list,truth_F,truth_T0,truth_T = data.values()
    dataloader,valid_loader,shuffle_dataloader,truth_category,_,_,_,train_dataset,_= create_dataloader(batch_size,file_name_option,valid_list)
    
    existing_file = glob.glob(os.path.join(model_dir,f"CSL_Module/iter=*_mutual_iteration={iter}.npy"))[0]
    data = np.load(existing_file,allow_pickle=True).item()
    truth_label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
    F = np.array(data["F"])
    z = data["z"]
    w = data["w"]
    lam = data["lam"]
    mu =data["mu"]
    theta = data["theta"]
    T = data["T"]
    T0 = data["T0"]
    pi = data["pi"]
    c = data["c"]
    ARI_list, ARI_index,EAR = calc_ARI_EAR(c,truth_category,F,truth_F)
    print(plot_category_image(result_MI_dir,train_dataset.image,c,ARI_index,3,3))
    print(lam)
    plt.figure()
    sns.heatmap(T)
    plt.savefig(os.path.join(result_MI_dir,f"Heatmap of T"))
    plt.close()
    sns.heatmap(theta)
    print("ARI a:",ARI_list,ARI_index,EAR)
    A = 10
    K = 10
    theta_a = []
    index = []
    category_num_dict = []
    for a in ARI_index:
        data_existing_category = np.where(np.bincount(c[:,a])>30)[0]
        category_list = []
        for k in range(K):            
            if k in data_existing_category:
                index.append(r"$a$"+f"={a}"+r",c"+f"={k}")
                theta_a.append(theta[a*K+k])
                category_list.append(k)
        category_num_dict.append(category_list)
    columns = word[:theta.shape[1]]
    theta = np.stack(theta_a)
    if len(theta.shape)==2:
        heat_theta = pd.DataFrame(theta,columns=columns,index=index)
    else:
        heat_theta = pd.DataFrame(np.reshape(theta,(-1,theta.shape[2])))
    truth_attribute_num_list = [6,6,6,4,4]
    chained_frame = pd.DataFrame(theta,columns=columns)
    heat_map_filepath = os.path.join(result_MI_dir,"Sorted heatmap of theta.pdf")
    diagonallike_heatmap(filepath=heat_map_filepath,dataframe=heat_theta,each_attribute_cats=category_num_dict,chained_frame=chained_frame,truth_attribute_num_list=truth_attribute_num_list)
    
    truth_F = np.tile(np.arange(5),(F.shape[0],1))
    cw= np.stack((c[:,8][:20000],w[:,1][:20000])).T
    a = np.arange(9)[:4]

    np.set_printoptions(threshold=np.inf)
    '''for i in range(20):
        print(cw[i*1000:(i+1)*1000])
        print(i)
        plt.figure()
        #plt.show()'''
    
    for a in range(10):
        bins = np.linspace(np.min(z[:,a]),np.max(z[:,a]),100)
        plt.figure()
        
        for k in range(10):
            index = np.where(c[:,a]==k)[0]
            z_a_k = z[:,a][index]
            plt.hist(z_a_k,bins = bins,alpha=0.5,label=f"c={k+1}")
        plt.xlabel(r"$z$")
        plt.legend()
        plt.savefig(os.path.join(result_MI_dir,f"Histgram z_{a}"))

    for attri,a in enumerate(ARI_index):
        bins = np.linspace(np.min(z[:,a]),np.max(z[:,a]),100)
        plt.figure()
        for k in np.unique(truth_category[:,attri]):
            index = np.where(truth_category[:,attri]==k)[0]
            z_a_k = z[:,a][index]
            plt.hist(z_a_k,bins = bins,alpha=0.5,label=f"c={k+1}")
        plt.xlabel(r"$z$")
        plt.legend()
        plt.savefig(os.path.join(result_MI_dir,f"Truth_label_Histgram z_{a}"))