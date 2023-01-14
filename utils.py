import copy,os,math,glob,cv2,torch,itertools
from numbers import Number
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import Dataset_3dshapes
from betaVAE import betaVAE
from file_option_name_memo import file_name_option_dict,label_word_correspondance
from functools import reduce

def label_to_vocab(label):
    attribute_num = label.shape[1]
    index_offset = 0
    for a in range(attribute_num):
        for i, category in enumerate(np.unique(label[:,a])):
            index = np.where(label[:,a]==category)[0]
            label[:,a][index] =i
        label[:,a] += index_offset
        index_offset += len(np.unique(label[:,a]))
    w = label.astype(np.int8)
    return w

def label_to_word(file_name_option):
    attribute_num = len(label_word_correspondance)
    word_list = np.array(list(itertools.chain.from_iterable(label_word_correspondance)))
    word_list_len = len(word_list)
    len_list = [len(label_word_correspondance[a]) for a in range(attribute_num)]
    index_list = np.arange(word_list_len)
    index_offset = 0
    eliminate_list = []
    for a in range(attribute_num):
        eliminate_list.append(np.array(file_name_option_dict[file_name_option][a])+index_offset)
        index_offset += len_list[a]
    eliminate_list = np.concatenate(eliminate_list).astype(np.int8)
    index_list = np.delete(index_list,eliminate_list).astype(np.int8)
    word = word_list[index_list]
    return word

def save_model(model,loss_mode, latent_dim, beta, linear_dim, prior_logvar,i, file_name_option = None):
    if file_name_option != None:
        existfiles = glob.glob(f"model_{loss_mode}VAE/main/latent_dim={latent_dim}/beta={beta}_linear={linear_dim}_prior_logvar={prior_logvar}_epoch=*_{file_name_option}.pth")
        torch.save(model.state_dict(), f"model_{loss_mode}VAE/main/latent_dim={latent_dim}/beta={beta}_linear={linear_dim}_prior_logvar={prior_logvar}_epoch={i+1}_{file_name_option}.pth")
        for f in existfiles:
            os.remove(f)
    else:
        existfiles = glob.glob(f"model_{loss_mode}VAE/main/latent_dim={latent_dim}/beta={beta}_linear={linear_dim}_prior_logvar={prior_logvar}_epoch=*.pth")
        torch.save(model.state_dict(), f"model_{loss_mode}VAE/main/latent_dim={latent_dim}/beta={beta}_linear={linear_dim}_prior_logvar={prior_logvar}_epoch={i+1}.pth")
        for f in existfiles:
            os.remove(f)
        
def log_N_density(z,mu,logvar):
    inv_sigma = np.exp(-logvar)
    tmp = (z - mu) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * logvar + np.log(2*np.pi))

def estimate_entropies(z,mu,logvar,n_samples=300): #n_samplesはメモリ的に10000は無理  #Minibatch Weighted Sampling で近似
    z=z[:,np.random.permutation(z.shape[1])[:n_samples]] 
    K,S = z.shape
    N,_ = mu.shape  
    weights = - np.log(N)

    entropies = np.zeros(K)
    k=0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = log_N_density(np.repeat(z.reshape(1 ,K, S),N,axis=0)[:,:,k:k+batch_size],
                                np.repeat(mu.reshape(N,K,1),S,axis=2)[:,:,k:k+batch_size],
                                np.repeat(logvar.reshape(N,K,1),S,axis=2)[:,:,k:k+batch_size])
        k +=batch_size
        entropies += - logsumexp(logqz_i+weights,dim=0,keepdim=False).sum(1)
    entropies /= S
    return entropies

def logsumexp(value,dim=None, keepdim=False):
    if dim is not None:
        m = np.max(value, axis=dim,keepdims=True)
        value0 = value - m
        if keepdim is False:
            m = np.squeeze(m,axis=dim)
        return m+ np.log(np.sum(np.exp(value0),axis=dim,keepdims=keepdim))
    else:
        m = np.max(value)
        sum_exp = np.sum(np.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + np.log(sum_exp)
    

def HSV2RGB(HSV_img):
    HSV_img=HSV_img.to("cpu").detach().numpy().transpose(1,2,0)
    RGB_img=copy.deepcopy(HSV_img)
    H_max = 255
    S_max = 255
    V_max = 255
    RGB_img[:,:,0]=HSV_img[:,:,0]*H_max
    RGB_img[:,:,1]=HSV_img[:,:,1]*S_max
    RGB_img[:,:,2]=HSV_img[:,:,2]*V_max
    RGB_img = RGB_img.astype(np.uint8)
    RGB_img = cv2.cvtColor(RGB_img,cv2.COLOR_HSV2BGR_FULL)
    return RGB_img

def create_dataloader(batch_size,file_name_option = "full", valid_list = []):
    data_filename = f"dataset/3dshapes_hsv_images_{file_name_option}.npy"; label_filename = f"dataset/3dshapes_hsv_labels_{file_name_option}.npy"
    image = np.load(data_filename).transpose((0,3,1,2)); label = label_to_vocab(np.load(label_filename))    
    
    print("create dataloader")
    valid_index = np.array([],dtype=np.int32)
    for valid_word_sequence in valid_list:
        v_list =[]
        for valid_word in valid_word_sequence:
            v_list.append(np.where(label==valid_word)[0])
        valid_index = np.append(valid_index,reduce(np.intersect1d,v_list))
    n_samples = len(label)
    train_index = np.delete(np.arange(n_samples),valid_index)
    train_dataset = Dataset_3dshapes(image[train_index],label[train_index]);valid_dataset = Dataset_3dshapes(image[valid_index],label[valid_index]) 
    full_dataset = Dataset_3dshapes(image,label)
    train_shuffle_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=False)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size,shuffle=False)
    full_shuffle_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,shuffle=False)
    print("total number of data points",len(train_dataset))
    return train_loader,valid_loader,train_shuffle_loader,train_dataset.label[:,:5],full_dataloader,full_shuffle_loader

def VAE_model(beta,latent_dim,device,linear_dim,dataset_name):
    image_channel=3
    if dataset_name =="3dshapes":
        image_size = 64
    layer = 4
    en_channel=[image_channel,32,64,128,256]
    de_channel=[256,128,64,32,image_channel]
    en_kernel=np.repeat(4,layer)
    de_kernel=np.repeat(4,layer)
    en_stride=np.repeat(2,layer)
    de_stride=np.repeat(2,layer)
    en_padding=np.repeat(1,layer)
    de_padding=np.repeat(1,layer)
    return betaVAE(beta,latent_dim,linear_dim,  en_channel,de_channel,en_kernel,de_kernel,en_stride,de_stride,en_padding,de_padding,image_size).to(device)