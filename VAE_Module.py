import torch,os,glob,time,math,cv2
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import save_model,HSV2RGB,create_dataloader
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) # [,]
class UnFlatten(nn.Module):
    def __init__(self,flatten_dim,last_channel):
        super().__init__()
        self.flatten_dim = flatten_dim
        self.last_channel = last_channel
    def forward(self, input):
        return input.view(-1,self.last_channel, self.flatten_dim, self.flatten_dim)

class VAE_Module(nn.Module):
    def __init__(self,VAE_Module_parameters,file_name_option,valid_list):
        super(VAE_Module, self).__init__()
        #Setting Network Architecture
        self.dataloader,self.valid_loader,self.shuffle_dataloader,self.w,_,_,_,_,_= create_dataloader(VAE_Module_parameters["batch_size"],file_name_option,valid_list)
        self.beta,self.latent_dim,self.linear_dim,self.epoch,self.image_size,self.batch_size = VAE_Module_parameters.values()
        self.initial_beta = np.copy(self.beta)
        layer = 4
        image_channel=3
        en_channel=[image_channel,32,64,128,256]
        de_channel=[256,128,64,32,image_channel]
        en_kernel=np.repeat(4,layer)
        de_kernel=np.repeat(4,layer)
        en_stride=np.repeat(2,layer)
        de_stride=np.repeat(2,layer)
        en_padding=np.repeat(1,layer)
        de_padding=np.repeat(1,layer)
        self.D=self.w.shape[0]
        self.layer_num = layer
        self.last_channel = en_channel[-1]
        flatten_dim = int((self.image_size/(2**self.layer_num))*(self.image_size/(2**self.layer_num))*self.last_channel)
        self.normalization = torch.Tensor([np.log(2 * np.pi)])
        self.register_buffer('prior_mu', torch.zeros(self.D,self.latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(self.D,self.latent_dim))
        self.prior_mu.requires_grad = False
        self.prior_logvar.requires_grad = False
        modules=[]
        for h in range(len(en_channel)-1):
            modules.append(
                nn.Conv2d(in_channels=en_channel[h], out_channels=en_channel[h+1], kernel_size=en_kernel[h], stride=en_stride[h], padding=en_padding[h])
            )
            modules.append(
                nn.ReLU()
            )
        modules.append(Flatten())
        modules.append(nn.Linear(flatten_dim, self.linear_dim))
        self.encoder= nn.Sequential(*modules)
        
        self.fc1 = nn.Linear(self.linear_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.linear_dim, self.latent_dim)
        self.fc3 = nn.Linear(self.latent_dim, self.linear_dim)
        
        modules=[]
        H=len(de_channel)-1
        modules.append(nn.Linear(self.linear_dim,flatten_dim))
        modules.append(UnFlatten(int(self.image_size/(2**self.layer_num)),self.last_channel))
        for h in range(H):
            modules.append(
                nn.ConvTranspose2d(in_channels=de_channel[h], out_channels=de_channel[h+1], kernel_size=de_kernel[h], stride=de_stride[h], padding=de_padding[h])
            )
            if h != H-1:
                modules.append(
                    nn.ReLU()
                )
        modules.append(nn.Sigmoid())
        self.decoder= nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, o_d):
        h = self.encoder(o_d)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        recon = self.decoder(z)
        return recon

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, o_d):
        z, mu, logvar = self.encode(o_d)
        return self.decode(z), mu, logvar, z
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar,batch_prior_mu,batch_prior_logvar, mutual_iteration):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        if mutual_iteration==0:        
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() )
        else:
            batch_prior_mu.requires_grad = False
            batch_prior_logvar.requires_grad = False
            var_division = logvar.exp() / batch_prior_logvar.exp() # Σ_0 / Σ_1
            diff = mu - batch_prior_mu # μ_１ - μ_0            
            diff_term = diff *diff / batch_prior_logvar.exp() # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1            
            logvar_division = batch_prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)            
            KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.latent_dim).sum()
        return BCE + self.beta * KLD


    def learn(self,mutual_iteration,model_dir):
        model=copy.deepcopy(self.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        self.model_dir = model_dir
        loss_list = np.array([])
        #self.beta = self.initial_beta * (0.8)**mutual_iteration
        for i in range(self.epoch):
            train_loss = 0
            s=time.time()
            for data, _,batch_index in self.shuffle_dataloader:
                data = Variable(data).to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar, _ = model.forward(data)
                batch_prior_mu = self.prior_mu[batch_index]
                batch_prior_logvar = self.prior_logvar[batch_index]
                loss = model.loss_function(recon_batch, data, mu, logvar, batch_prior_mu,batch_prior_logvar ,mutual_iteration)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            loss_list = np.append(loss_list,train_loss / self.D)
            if i==0 or (i+1) % (self.epoch // 5) == 0 or i == (self.epoch-1):
                print('====> Beta: {} Epoch: {} Average loss: {:.4f}  Learning time:{}s'.format(np.round(self.beta,2), i+1, train_loss / self.D,np.round(time.time()-s,2)))
                self.save_model(model,mutual_iteration,i+1)
        z_list = []
        for data,_,_ in self.dataloader:
            data=data.to(device)
            _,_,_,batch_z = model.forward(data)
            z_list.append(batch_z.to("cpu").detach().numpy())
        z = np.concatenate(z_list)
        return z

    def save_model(self,model,mutual_iteration,i):
        existfiles = glob.glob(os.path.join(self.model_dir,f"mutual_iteration={mutual_iteration}_epoch=*.pth"))
        torch.save(model.state_dict(), os.path.join(self.model_dir,f"mutual_iteration={mutual_iteration}_epoch={i}.pth"))
        for f in existfiles:
            os.remove(f)