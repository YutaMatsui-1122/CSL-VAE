from VAE_Module import VAE_Module
from CSL_Module import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob,os
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class CSL_VAE():
    def __init__(self,file_name_option,mutual_iteration_number,CSL_Module_parameters,VAE_Module_parameters,valid_list,grammar_list=[[0,1,2,3,4]],Nd_rate=[0,0,0,0,1]) -> None:
        self.VAE_Module_parameters = VAE_Module_parameters
        self.CSL_Module_parameters = CSL_Module_parameters
        self.file_name_option = file_name_option
        self.valid_list = valid_list
        self.grammar_list = grammar_list
        self.Nd_rate = Nd_rate
        self.vae_module = VAE_Module(VAE_Module_parameters,self.file_name_option,self.valid_list)
        self.csl_module = CSL_Module(CSL_Module_parameters)
        self.w = self.vae_module.w.to('cpu').detach().numpy().copy()
        self.truth_category = self.w.copy()
        self.D,_ = self.w.shape
        self.create_word_sequence(grammar_list,Nd_rate)        
        self.csl_module.initialize_parameter(self.w,np.zeros((self.D,self.vae_module.latent_dim)),self.N)
        self.mutual_iteration_number = mutual_iteration_number
        self.phi = []

    def experimental_setting(self):
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
        f = open(os.path.join(self.exp_dir,'Experimental_Setting.txt'), 'w', newline='\n')
        f.write(f"Global parapeters\n")
        f.write(f"  Mutual Iteration".ljust(20)+f" : {self.mutual_iteration_number}"+"\n")
        f.write(f"  file name option".ljust(20)+f" : {self.file_name_option}"+"\n")
        f.write(f"  valid list".ljust(20)+f" : {self.valid_list}"+"\n")
        f.write(f"  grammar list".ljust(20)+f" : {self.grammar_list}"+"\n")
        f.write(f"  Nd rate".ljust(20)+f" : {self.Nd_rate}"+"\n")
        f.write(f"\nCSL Module parapeters\n")
        for key,value in self.CSL_Module_parameters.items():
            f.write(f"  {key}".ljust(20)+f" : {value}"+"\n")
        f.write(f"\nVAE Module parapeters\n")
        for key,value in self.VAE_Module_parameters.items():
            f.write(f"  {key}".ljust(20)+f" : {value}"+"\n")
        f.close()
        print(f"Experiment {len(existing_exp_dir)}")

    def create_word_sequence(self,grammar_list,Nd_rate):
        #create word sequence
        # grammar_list : designate the order of attributes
        # Nd_rate      : designate the rate of word sequence length
        N_max = len(Nd_rate)
        '''self.truth_T0 = np.array([0.5,0.2,0.3,0,0])
        self.truth_T = np.array([[0,0.6,0.2,0.2,0,0],[0,0,0.5,0.3,0.1,0.1],[0,0,0,0.4,0.4,0.2],[0,0,0,0,0.7,0.3],[0,0,0,0,0,1],[0,0,0,0,0,1]])'''
        self.truth_T0 = np.array([1,0,0,0,0])
        self.truth_T = np.array([[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0,0,0,1]])
        F_list = np.array(list(itertools.permutations(range(N_max),N_max)))
        T_hat = [self.truth_T0[F[0]] * np.prod([self.truth_T[F[n-1]][F[n]] for n in range(1,N_max)])  for F in F_list]
        T_hat /= np.sum(T_hat)
        grammer_num = len(grammar_list)
        D = len(self.w)
        self.truth_F = np.zeros_like(self.w,dtype=np.int8)
        self.N = np.zeros(self.D,dtype=np.int8)
        for d in range(D):
            self.truth_F[d][0] = np.random.choice(N_max,p=self.truth_T0)
            self.w[d][0] = self.w[d][self.truth_F[d][0]]
            self.N[d] += 1
            for n in range(1,N_max):
                self.truth_F[d][n] = np.random.choice(N_max+1,p=self.truth_T[self.truth_F[d][n-1]])
                if self.truth_F[d][n] == N_max:
                    self.w[d][n] = -1
                else:
                    self.w[d][n] = self.w[d][self.truth_F[d][n]]
                    self.N[d] += 1
                    
    def setting_learned_model(self,w,beta,file_name_option,mutual_iteration,model_dir,valid_list,grammar_list,Nd_rate):
        model_file = glob.glob(os.path.join(model_dir,"VAE_Module",f"mutual_iteration={mutual_iteration}_epoch=*.pth"))[0]
        self.vae_module.load_state_dict(torch.load(model_file))
        self.csl_module.setting_parameters(w,z=np.zeros((self.D,self.VAE_Module_parameters["latent_dim"])),N_list=self.N,mutual_iteration=mutual_iteration,model_dir=model_dir)

    def learn(self):
        self.experimental_setting()
        reset_CSL = True
        word_sequence_info = {"valid_list":self.valid_list,"grammar_list":self.grammar_list,"Nd_rate":self.Nd_rate,"truth_F":self.truth_F,"truth_T0":self.truth_T0,"truth_T":self.truth_T}
        np.save(os.path.join(self.exp_dir,"word sequence setting"),word_sequence_info,allow_pickle=True)
        for i in range(self.mutual_iteration_number):
            print(f"Start Mutural Iteration {i}")
            debug = False
            if debug :
                model_file = glob.glob(os.path.join(self.root_dir,"exp19/model/VAE_Module",f"mutual_iteration={0}_epoch=*.pth"))[0]
                self.vae_module.load_state_dict(torch.load(model_file))
                self.vae_module = self.vae_module.to(device)
                z_list = []
                for data,_,_ in self.vae_module.dataloader:
                    data=data.to(device)
                    _,_,_,batch_z = self.vae_module.forward(data)
                    z_list.append(batch_z.to("cpu").detach().numpy())
                z = np.concatenate(z_list)
            else:
                z=self.vae_module.learn(i,self.vae_module_dir)
            phi_hat = self.csl_module.learn(self.w,z,i,self.csl_module_dir,self.N,self.truth_category,reset_parameter=reset_CSL)
            self.vae_module.prior_mu=torch.from_numpy(phi_hat[0]).clone().to(device)
            self.vae_module.prior_logvar=torch.from_numpy(phi_hat[1]).clone().to(device).log()
        
    def img2wrd(self,o_star,EOS):
        o_star = torch.unsqueeze(o_star,0)
        z_star,_,_ = self.vae_module.encode(o_star)
        z_star = z_star.detach().numpy()[0]
        c_star = self.csl_module.img2wrd_sampling_c(z_star,sampling_flag=False)
        F_star = self.csl_module.img2wrd_sampling_F(EOS)
        w_star = self.csl_module.img2wrd_sampling_w(c_star,F_star)
        return w_star

    def wrd2img(self,w_star):
        F_star = self.csl_module.wrd2img_sampling_F(w_star)
        c_star = self.csl_module.wrd2img_sampling_c(w_star, F_star)
        #c_star = self.csl_module.wrd2img_sampling_c_joint_F(w_star,sampling_flag=False)
        z_star = self.csl_module.wrd2img_sampling_z(c_star,sampling_flag=False)
        z_star = torch.from_numpy(z_star).float()
        z_star = torch.unsqueeze(z_star,0)
        o_star = self.vae_module.decode(z_star)[0]
        return o_star

batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "five_view_point_55544"
dup_num = 2
shift_rate = 0.01
Attribute_num = 5
file_name_option += f"×{dup_num}_shift_{shift_rate}"
label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
w=label_to_vocab(label[:,:5])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Experimenttal Setting
mutual_iteration_number = 15
valid_list = [[3,6,11,16,20],[0,5,14,17,22],[4,7,12,16,19],[1,9,10,18,21]]
grammar_list = [[0,1,2,3,4],[1,0,2,3,4],[0,1,3,2,4],[1,0,3,2,4],[2,3,4,0,1],[2,3,4,1,0],[3,2,4,0,1],[3,2,4,1,0]]
latent_dim = 10
CSL_Module_parameters = {"MAXITER":100,"A":latent_dim,"K":10,"alpha_T":1,"alpha_theta":0.1,"alpha_T0":1,"alpha_pi":1,"m":0,"tau":0.01,"a_lam":10,"b_lam":10}
VAE_Module_parameters = {"beta":20,"latent_dim":latent_dim,"linear_dim":1024,"epoch":300,"image_size":64,"batch_size":500}
#Nd_rate = [0, 0, 0.2, 0.3, 0.5]
Nd_rate = [0, 0, 0, 0, 1]

if __name__ == "__main__":
    csl_vae = CSL_VAE(file_name_option,mutual_iteration_number,CSL_Module_parameters,VAE_Module_parameters,valid_list,grammar_list,Nd_rate)
    csl_vae.learn()