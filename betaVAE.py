import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.autograd import Variable
import cv2
import math
import time   
from scipy.stats import bernoulli
import sys

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

class betaVAE(nn.Module):
    def __init__(self,beta,latent_dim,linear_dim,en_channel,de_channel,en_kernel,de_kernel,en_stride,de_stride,en_padding,de_padding,image_size):
        super(betaVAE, self).__init__()
        self.beta=beta
        self.latent_dim=latent_dim
        self.layer_num = len(en_kernel)
        self.last_channel = en_channel[-1]
        flatten_dim = int((image_size/(2**self.layer_num))*(image_size/(2**self.layer_num))*self.last_channel)
        self.normalization = torch.Tensor([np.log(2 * np.pi)])
        self.register_buffer('prior_mu', torch.zeros(self.latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(self.latent_dim))
        modules=[]
        for h in range(len(en_channel)-1):
            modules.append(
                nn.Conv2d(in_channels=en_channel[h], out_channels=en_channel[h+1], kernel_size=en_kernel[h], stride=en_stride[h], padding=en_padding[h])
            )
            modules.append(
                nn.ReLU()
            )
        modules.append(Flatten())
        modules.append(nn.Linear(flatten_dim, linear_dim))
        self.encoder= nn.Sequential(*modules)
        
        self.fc1 = nn.Linear(linear_dim, latent_dim) 
        self.fc2 = nn.Linear(linear_dim, latent_dim)  
        self.fc3 = nn.Linear(latent_dim, linear_dim)
        
        modules=[]
        H=len(de_channel)-1
        modules.append(nn.Linear(linear_dim,flatten_dim))
        modules.append(UnFlatten(int(image_size/(2**self.layer_num)),self.last_channel))
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
        z = self.decoder(z)
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, o_d):
        z, mu, logvar = self.encode(o_d)
        return self.decode(z), mu, logvar, z
    
    def logsumexp(self,value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(torch.exp(value0),
                                        dim=dim, keepdim=keepdim))
        
    def log_density(self, z, mu, logvar):
        import math
        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-logvar)
        tmp = z - mu
        return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)
    
    def Total_correlation(self, z, mu, logvar):
        logqz_prob = self.log_density(z.unsqueeze(dim=1), mu.unsqueeze(dim=0), logvar.unsqueeze(dim=0))
        
        logqz_product = logqz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)
        logqz = logqz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()
        return (logqz - logqz_product).mean()
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar,z,prior_mu, prior_logvar, loss_mode="beta", prior_logvar_value = 0,mutual_learning_flag = False):
        prior_logvar = torch.full_like(mu,prior_logvar_value)
        if loss_mode == "beta":
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) / prior_logvar.exp() - logvar.exp() / prior_logvar.exp() -prior_logvar)
            #print(KLD.item(), (p_z_x.log() - p_z.log()).sum().item(), (self.ln_q(p_z_x,0.99999) - self.ln_q(p_z,0.99999)).sum().item())
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            return BCE + self.beta * KLD
        elif loss_mode == "beta-TC":
            if mutual_learning_flag:
                mu_diff = mu-prior_mu
                KLD = -0.5 * torch.sum(1 + logvar - prior_logvar - logvar.exp() / prior_logvar.exp() - mu_diff.pow(2) / prior_logvar.exp())
            else: 
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) / self.prior_logvar.exp() - logvar.exp() / self.prior_logvar.exp() -self.prior_logvar)
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')            
            TC=self.Total_correlation(z, mu, logvar)
            #print(((logqz_condx - logqz) -  self.beta * (logqz - logqz_prodmarginals) - (logqz_prodmarginals - logpz)).mean().mul(-1))
            return BCE + self.beta * KLD + (1 - self.beta) * TC
        elif loss_mode == "q":
            q=0.9;q_dash=0.999;beta=3500;gamma=3500
            #print((1-q)*(1-q_dash)**(-1))
            p_z = self.multivariate_normal_distribution_diagonal(z,torch.zeros_like(mu),torch.zeros_like(logvar))
            p_z_x = self.multivariate_normal_distribution_diagonal(z,mu,logvar)
            ln_q_p_x_z = self.ln_q_bernoulli(x,recon_x,q_dash)
            loss = -(p_z**(1-q) * ln_q_p_x_z).sum() + beta * self.ln_q(p_z_x,q).sum() - gamma * self.ln_q(p_z,q).sum()
            #loss = -(p_z**(1-q) * ln_q_p_x_z + beta * self.ln_q(p_z,q) - gamma * self.ln_q(p_z_x,q))
            #print(loss.sum())
            #print(torch.count_nonzero(torch.isnan(p_z**(1-q) * ln_q_p_x_z)).item(),torch.count_nonzero(torch.isnan(beta * self.ln_q(p_z,q))).item(),torch.count_nonzero(torch.isnan(gamma * self.ln_q(p_z_x,q))).item())
            #print(torch.count_nonzero(torch.isnan(p_z)).item(),torch.count_nonzero(torch.isnan(p_z_x)).item(),torch.count_nonzero(torch.isnan(ln_q_p_x_z)).item())
            return loss

    def multivariate_normal_distribution_diagonal(self,z, mu, logvar):
        z_mu = z-mu
        return (-0.5*(z_mu**2 / logvar.exp() + logvar + math.log(2*math.pi))).exp().prod(1)
    
    def KL_q(self,mu,logvar,q):
        var = logvar.exp()
        self.latent_dim
        prior_mu = torch.zeros(self.latent_dim).to(device)
        mu_mu = prior_mu - mu
        prior_var = torch.ones(self.latent_dim).to(device)
        Sigma = q * prior_var + (1-q) * var
        I_q = (prior_var.prod()**q * var.prod(1)**(1-q)) / Sigma.prod(1) + q * (1-q) * (mu_mu**2/ Sigma).prod(1)
        KL = (torch.exp(0.5*I_q) - 1) / (q-1)
        return KL

    def ln_q_bernoulli(self,x,lam,q):
        #print(((1-q) * torch.log(self.Bernoulli(x,lam)))) #←ここで0になってるよ
        #print((((1-q) * self.Bernoulli(x,lam).log( ).sum((1,2,3))).exp() -1 ) / (1-q),(((1-q) * (-F.binary_cross_entropy(lam,x,reduction='none')).sum((1,2,3))).exp() -1 ) / (1-q))
        #print(self.ln_q(self.Bernoulli(x,lam),q).sum((1,2,3)).sum().item() == -)
        result = self.ln_q((-F.binary_cross_entropy(lam,x,reduction="none")).exp(),q).sum((1,2,3))
        return result
        return -F.binary_cross_entropy(lam,x,reduction="none").sum((1,2,3))
        return (((1-q) * self.Bernoulli(x,lam).log( ).sum((1,2,3))).exp() -1 ) / (1-q)
        return (((1-q) * (-F.binary_cross_entropy(lam,x,reduction='none')).sum((1,2,3))).exp() -1 ) / (1-q)

    def Bernoulli(self,x,lam):
        return (lam ** x * (1-lam) ** (1-x))

    def ln_q(self,x,q=1):
        '''
        Parameters
            q -> int
            x -> numpy.array
        
        Reference:
            Kobayashis, Taisuke. 
            "q-VAE for Disentangled Representation Learning and Latent Dynamical Systems." 
            IEEE Robotics and Automation Letters 5.4 (2020): 5669-5676.
        '''

        if q == 1:
            result = torch.log(x)
            return result
            
        else:
            return (x**(1-q) - 1) / (1-q)