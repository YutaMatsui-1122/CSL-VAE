from VAE_Module import VAE_Module
from CSL_Module import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob,os
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class CSL_VAE():
    def __init__(self) -> None:
        pass
    def initialize_model(self,beta,file_name_option,mutual_iteration_number,valid_list,grammar_list=[[0,1,2,3,4]],Nd_rate=[0,0,0,0,1]) -> None:
        self.mutual_iteration_number = mutual_iteration_number
        self.phi = []
        self.beta = beta
        self.epoch = 1000
        self.latent_dim = 10
        self.linear_dim = 1024
        self.image_size = 64
        self.batch_size = 500
        self.file_name_option = file_name_option
        self.valid_list = valid_list
        self.dataloader,self.valid_loader,self.shuffle_dataloader,w= create_dataloader(self.batch_size,self.file_name_option,self.valid_list)
        self.w = w.to('cpu').detach().numpy().copy()
        self.D = w.shape[0]
        self.truth_category = w.to('cpu').detach().numpy().copy()
        self.create_word_sequence(grammar_list,Nd_rate)
        self.vae_module = VAE_Module(self.beta,self.epoch,self.latent_dim,self.linear_dim,self.image_size,self.batch_size,self.w.shape[0],self.dataloader,self.shuffle_dataloader)
        self.csl_module = CSL_Module()
    
    def get_model_dir(self):
        self.root_dir = "exp_CSL_VAE"
        os.makedirs(self.root_dir,exist_ok=True)
        existing_exp_dir = glob.glob(os.path.join(self.root_dir,"exp*"))
        self.exp_dir = os.path.join(self.root_dir,f"exp{len(existing_exp_dir)}")
        self.model_dir = os.path.join(self.exp_dir,"model")
        self.csl_module_dir = os.path.join(self.model_dir,"CSL_Module")
        self.vae_module_dir = os.path.join(self.model_dir,"VAE_Module")
        os.makedirs(self.exp_dir,exist_ok=True)
        os.makedirs(self.model_dir,exist_ok=True)
        os.makedirs(self.csl_module_dir,exist_ok=True)
        os.makedirs(self.vae_module_dir,exist_ok=True)
        print(f"Experiment {len(existing_exp_dir)}")

    def create_word_sequence(self,grammar_list,Nd_rate):
        #create word sequence
        # grammar_list : designate the order of attributes
        # Nd_rate      : designate the rate of word sequence length 
        grammer_num = len(grammar_list)
        D = len(self.w)        
        for d in range(D):
            self.w[d] = self.w[d][grammar_list[d%grammer_num]]
        perm_w = np.random.permutation(self.w)
        Nd_list = [int(sum(Nd_rate[:i+1])*D) for i in range(len(Nd_rate))]
        new_w = []
        for i in range(5):
            w_i = np.split(perm_w,Nd_list)[i]
            w_i[:,i+1:] = -1
            new_w.append(w_i)

    def setting_learned_model(self,w,beta,file_name_option,mutual_iteration,model_dir,valid_list,grammar_list,Nd_rate):
        self.initialize_model(beta,file_name_option,mutual_iteration,valid_list,grammar_list,Nd_rate)
        model_file = glob.glob(os.path.join(model_dir,"VAE_Module",f"beta={beta}_mutual_iteration={mutual_iteration}_epoch=*.pth"))[0]
        self.vae_module.load_state_dict(torch.load(model_file))
        self.csl_module.setting_parameters(w,z=np.zeros((self.D,self.latent_dim)),mutual_iteration=mutual_iteration,model_dir=model_dir,K=10)
    
    def learn(self,beta,file_name_option,mutual_iteration_number,valid_list,grammar_list,Nd_rate):
        self.initialize_model(beta,file_name_option,mutual_iteration_number,valid_list,grammar_list,Nd_rate)
        self.get_model_dir()
        word_sequence_info = {"valid_list":valid_list,"grammar_list":grammar_list,"Nd_rate":Nd_rate}
        np.save(os.path.join(self.exp_dir,"word sequence setting"),word_sequence_info,allow_pickle=True)
        for i in range(self.mutual_iteration_number):
            debug = False
            if debug :
                model_file = glob.glob(os.path.join(self.root_dir,"exp1/model/VAE_Module",f"beta={beta}_mutual_iteration={0}_epoch=*.pth"))[0]
                self.vae_module.load_state_dict(torch.load(model_file))
                self.vae_module = self.vae_module.to(device)
                z_list = []
                for data,_,_ in self.dataloader:
                    data=data.to(device)
                    _,_,_,batch_z = self.vae_module.forward(data)
                    z_list.append(batch_z.to("cpu").detach().numpy())
                z = np.concatenate(z_list)
            else:
                z=self.vae_module.learn(i,self.vae_module_dir)
            phi_hat = self.csl_module.learn(self.w,z,i,self.csl_module_dir,self.truth_category)
            self.vae_module.prior_mu=torch.from_numpy(phi_hat[0]).clone().to(device)
            self.vae_module.prior_logvar=torch.from_numpy(phi_hat[1]).clone().to(device).log()
        
    def img2wrd(self,o_star,N_star):
        o_star = torch.unsqueeze(o_star,0)
        z_star,_,_ = self.vae_module.encode(o_star)
        z_star = z_star.detach().numpy()[0]
        c_star = self.csl_module.img2wrd_sampling_c(z_star)
        F_star = self.csl_module.img2wrd_sampling_F(N_star)
        w_star = self.csl_module.img2wrd_sampling_w(c_star,F_star,N_star)
        return w_star

    def wrd2img(self,w_star):
        F_star = self.csl_module.wrd2img_sampling_F(w_star)
        c_star = self.csl_module.wrd2img_sampling_c(w_star, F_star)
        z_star = self.csl_module.wrd2img_sampling_z(c_star,sampling_flag=False)
        z_star = torch.from_numpy(z_star).float()
        z_star = torch.unsqueeze(z_star,0)
        o_star = self.vae_module.decode(z_star)[0]
        return o_star

if __name__ == "__main__":
    batch_size = 500
    file_name_option = None
    dataset_name = "3dshapes"
    file_name_option = "three_view_point_88844"
    dup_num = 1
    shift_rate = 0.01
    Attribute_num = 5
    #file_name_option += f"×{dup_num}_shift_{shift_rate}"
    label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
    w=label_to_vocab(label[:,:5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    beta = 16
    latent_dim=10
    linear_dim=1024
    epoch=1000
    image_size = 64
    mutual_iteration_number = 6
    valid_list = [[5,8,20,25,30],[2,12,19,27,31],[7,14,23,27,28],[0,10,22,27,29]]
    grammar_list = [[0,1,2,3,4],[2,3,4,1,0],[3,2,4,0,1],[1,0,2,3,4]]
    Nd_rate = [0, 0.2, 0.2, 0.2, 0.4]
    Nd_rate = [0, 0, 0, 0, 1]
    csl_vae = CSL_VAE()
    #csl_vae.initialize_model(beta,file_name_option,mutual_iteration_number,valid_list,grammar_list,Nd_rate)

    csl_vae.learn(beta,file_name_option,mutual_iteration_number,valid_list,grammar_list,Nd_rate)