import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import glob
import os
import math
from utils import *
from Base_Model import BaseModel

def calc_ARI(label,truth_labels):
  ARI_list = []
  ARI_index = []
  for a in range(5):
    ARI_list.append(np.max([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
    ARI_index.append(np.argmax([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
  return np.round(ARI_list,decimals=3), ARI_index

def all_combination(N):
  all_comb = []
  #all_comb.append([])
  for n in range(1,N+1):
    for conb in itertools.combinations(np.arange(N), n):
      all_comb.append(list(conb))
  return all_comb

class CSL_Module():
  def __init__(self,CSL_Module_parameters) -> None:
    self.MAXITER,self.A,self.K,self.alpha_T,self.alpha_theta,self.alpha_T0,self.alpha_pi,self.m,self.tau,self.a_lam,self.b_lam = CSL_Module_parameters.values()

  def initialize_parameter(self,w,z,N_list):  
    self.w = w
    self.z = z
    self.D,self.N_max = self.w.shape
    self.V = np.max(self.w)+1
    self.L = self.A * self.K
    self.N = N_list
    self.F = np.random.randint(0,self.A,size = (self.D,self.N_max))
    self.F = np.array([[np.random.randint(0,self.A) if n<self.N[d] else self.A for n in range(self.N_max)]for d in range(self.D)])
    self.T0 = np.random.dirichlet(np.repeat(self.alpha_T0,self.A))
    self.T = np.random.dirichlet(np.repeat(self.alpha_T,self.A+1),size = self.A)
    self.theta = np.random.dirichlet(np.repeat(self.alpha_theta, self.V),size =self.L)
    self.pi = np.random.dirichlet(np.repeat(self.alpha_pi,self.K),size = self.A)
    self.lam = np.full(shape=(self.A,self.K),fill_value=10.0)
    self.mu = np.array([np.linspace(np.min(self.z[:,a]),np.max(self.z[:,a]),self.K) for a in range(self.A)])
    self.c = np.array([np.random.choice(self.K,p=self.pi[a],size = self.D) for a in range(self.A)]).T

  def receive_z(self,z):
    self.c = np.array([np.random.choice(self.K,p=self.pi[a],size = self.D) for a in range(self.A)]).T
    self.F = np.array([[np.random.randint(0,self.A) if n<self.N[d] else self.A for n in range(self.N_max)]for d in range(self.D)])
    self.z = z
  
  def set_parameters_for_economizing(self):
    self.sum_theta_aw = np.array([[np.sum(self.theta[a*self.K:(a+1)*self.K][:,w]) for w in range(self.V)] for a in range(self.A)] )
    self.F_list = [list(itertools.permutations(range(self.A),N_star)) for N_star in range(self.N_max+1)]
  
  def setting_parameters(self,w,z,N_list,mutual_iteration,model_dir):
    self.initialize_parameter(w,z,N_list)
    parameter_file = glob.glob(f"{model_dir}/CSL_Module/iter=*_mutual_iteration={mutual_iteration}.npy")[0]
    parameters = np.load(parameter_file,allow_pickle=True).item()
    _,_,_,self.T0,self.T,self.theta,self.pi,self.mu,self.lam,_ = parameters.values()
    self.set_parameters_for_economizing()

  def sampling_F(self):
    for d in range(self.D):
      theta_index_per_a = [a*self.K+self.c[d][a] for a in range(self.A)]
      if self.N[d]>1:
        T0_hat = self.T0 * self.T[:,self.F[d][1]] * self.theta[theta_index_per_a,self.w[d][0]]
      else:
        T0_hat = self.T0 * self.theta[theta_index_per_a,self.w[d][0]]
      T0_hat /= np.sum(T0_hat)
      self.F[d][0] = np.random.choice(self.A,p=T0_hat)
      for n in range(1,self.N[d]):
        if n == (self.N[d]-1):
          theta_hat = self.T[self.F[d][n-1]][:self.A] * self.theta[theta_index_per_a,self.w[d][n]]
          theta_hat /= np.sum(theta_hat)
          self.F[d][n] = np.random.choice(self.A,p=theta_hat)
        else:
          theta_hat = self.T[self.F[d][n-1]][:self.A] * self.T[:,self.F[d][n+1]] * self.theta[theta_index_per_a,self.w[d][n]]
          theta_hat /= np.sum(theta_hat)
          self.F[d][n] = np.random.choice(self.A,p=theta_hat)

  def sampling_T0(self):
    alpha_T0_hat = np.bincount(self.F[:,0],minlength=self.A) + self.alpha_T0
    self.T0 = np.random.dirichlet(alpha_T0_hat)

  def sampling_T(self):
    for a in range(self.A):
      index_d,index_n = np.where(self.F[:,:self.N_max-1]==a)
      index_n+=1
      alpha_T_hat = np.bincount(self.F[index_d,index_n],minlength=self.A+1) + self.alpha_T
      self.T[a] = np.random.dirichlet(alpha_T_hat)  

  def sampling_theta(self):
    alpha_theta_hat = np.full((self.L,self.V),self.alpha_theta)
    for d in range(self.D):
      for n in range(self.N[d]):
        alpha_theta_hat[self.F[d][n]*self.K+self.c[d][self.F[d][n]]][self.w[d][n]] += 1
    for l in range(self.L):
      self.theta[l] = np.random.dirichlet(alpha_theta_hat[l])
    
  def sampling_pi(self):
    for a in range(self.A):
      alpha_pi_hat = np.bincount(self.c[:,a],minlength=self.K) + self.alpha_pi
      self.pi[a] = np.random.dirichlet(alpha_pi_hat)

  def sampling_mu_lam(self):
    for a in range(self.A):
      z = self.z[:,a][:,np.newaxis]
      c = np.identity(self.K)[self.c[:,a]]     
      tau_hat=np.sum(c,axis=0)+self.tau
      mu_hat=(np.sum(c*z,axis=0) + self.tau * self.m) / tau_hat
      a_hat = 0.5 * np.sum(c,axis=0) + self.a_lam
      b_hat = 0.5 * (np.sum(c*z**2,axis=0) + self.tau * self.m ** 2 - tau_hat * mu_hat ** 2) + self.b_lam
      for k in range(self.K):
          #sampling mu and lambda
          self.lam[a][k]=np.random.gamma(a_hat[k],1/b_hat[k],1)
          self.mu[a][k]=np.random.normal(mu_hat[k],1/(tau_hat[k]*self.lam[a][k])**0.5)

  def sampling_c(self):
    log_theta = np.log(self.theta)
    log_pi = np.log(self.pi)    
    for a in range(self.A):
      logpdf_kd = np.array([self.logpdf(self.z[:,a],self.mu[a][k],self.lam[a][k]) for k in range(self.K)]).T
      for d in range(self.D):
        pi_hat = np.zeros(self.K,dtype=np.float128)
        pi_hat += log_pi[a]
        for n in np.where(self.F[d]==a)[0]:
          pi_hat += log_theta[a*self.K:(a+1)*self.K,self.w[d][n]]
        pi_hat += logpdf_kd[d]
        pi_hat = np.exp(pi_hat).astype(np.float64)
        pi_hat /= np.sum(pi_hat)
        self.c[d][a] = np.random.choice(self.K,p=pi_hat)
  
  def img2wrd_sampling_c(self,z_star,sampling_flag = True):
    c_star = np.zeros(self.A,dtype=np.int8)
    log_pi = np.log(self.pi)
    for a in range(self.A):
      log_pi_hat = np.zeros(self.K,np.float64)
      log_pi_hat += log_pi[a]
      log_pi_hat += np.array([self.logpdf(z_star[a],self.mu[a][k],self.lam[a][k]) for k in range(self.K)])
      pi_hat = np.exp(log_pi_hat)
      pi_hat /= np.sum(pi_hat)
      if sampling_flag:
        c_star[a] = np.random.choice(self.K,p=pi_hat)
      else:
        c_star[a] = np.argmax(pi_hat)
    return c_star
  
  def img2wrd_sampling_F(self,sampling_method = "sequential"):
    #F_list = np.array(list(itertools.product(range(self.A),repeat=N_star)))
    if type(sampling_method)==int:
      N_star = sampling_method
      T_hat = [self.T0[F[0]] * np.prod([self.T[F[n-1]][F[n]] for n in range(1,N_star)])  for F in self.F_list[N_star]]
      T_hat /= np.sum(T_hat)
      F_star = self.F_list[N_star][np.random.choice(len(T_hat),p=T_hat)]

    elif sampling_method == "sequential":
      F_star = []
      F_star.append(np.random.choice(self.A,p= self.T0))
      for n in range(1,self.N_max):
        F_star_n = np.random.choice(self.A+1,p= self.T[F_star[n-1]])
        if F_star_n==self.A: #if EOS
          break
        F_star.append(F_star_n)
      F_star = np.array(F_star)
      
    '''F_list = np.array(list(itertools.permutations(range(self.A),N_star)))
    T_hat = [self.T0[F[0]] * np.prod([self.T[F[n-1]][F[n]] for n in range(1,N_star)])  for F in F_list]
    T_hat /= np.sum(T_hat)
    F_star = F_list[np.random.choice(len(T_hat),p=T_hat)]
    F_star = F_list[np.argmax(T_hat)]'''
    return F_star

  def img2wrd_sampling_w(self,c_star,F_star):
    N_star = len(F_star)
    w_star = np.zeros(N_star,dtype=np.int8)
    for n in range(N_star):
      w_star[n] = np.random.choice(self.V,p=self.theta[F_star[n]*self.K + c_star[F_star[n]]])
    return w_star

  def wrd2img_sampling_F(self,w_star,sampling_method = "simultaneous"):
    N_star = len(w_star)
    if sampling_method == "simultaneous":
      T_hat = [np.prod([self.sum_theta_aw[F[n]][w_star[n]] for n in range(N_star)]) * self.T0[F[0]] * np.prod([self.T[F[n-1]][F[n]] for n in range(1,N_star)])  for F in self.F_list[N_star]]
      T_hat /= np.sum(T_hat)
      F_star = self.F_list[N_star][np.random.choice(len(T_hat),p=T_hat)]
      F_star = np.array(self.F_list[N_star][np.argmax(T_hat)])
    elif sampling_method == "sequential":
      F_star = np.zeros(N_star,dtype=np.int8)
      T0_hat = [self.sum_theta_aw[a][w_star[0]]*self.T0[a] for a in range(self.A)]
      T0_hat /= np.sum(T0_hat)
      F_star[0] = np.random.choice(self.A,p=T0_hat)
      F_star[0] = np.argmax(T0_hat)    
      for n in range(1,N_star):
        T_hat = [self.sum_theta_aw[a][w_star[n]]*self.T[F_star[n-1]][a] for a in range(self.A)]
        F_star[n] = np.argmax(T_hat)
    return F_star
    
  def wrd2img_sampling_c_joint_F(self,w_star,sampling_flag=True):
    N_star = len(w_star)
    c_star = np.zeros(self.A,dtype=np.int8)
    N_comb = all_combination(N_star)
    for a in range(self.A):
      pi_hat = np.zeros(self.K)
      theta_a = self.theta[a*self.K:(a+1)*self.K]
      for k in range(self.K):
        pi_hat[k] = np.sum([np.prod(theta_a[k][w_star[Ns]]) for Ns in N_comb]) * self.pi[a][k]
      pi_hat /= np.sum(pi_hat)
      if sampling_flag:
        c_star[a] = np.random.choice(self.K,p=pi_hat)
      else:
        c_star[a] = np.argmax(pi_hat)
    return c_star

  def wrd2img_sampling_c(self,w_star, F_star):
    c_star = np.zeros(self.A,dtype=np.int8)
    log_pi = np.log(self.pi)
    for a in range(self.A):
      pi_hat = np.zeros(self.K,dtype=np.float128)
      pi_hat += log_pi[a]
      for n in np.where(F_star==a)[0]:
        pi_hat += np.log(self.theta[a*self.K:(a+1)*self.K,w_star[n]])
      pi_hat = np.exp(pi_hat).astype(np.float64)
      pi_hat /= np.sum(pi_hat)
      #print(a,pi_hat)
      c_star[a] = np.random.choice(self.K,p=pi_hat)
    return c_star

  def wrd2img_sampling_z(self,c_star,sampling_flag = True):
    z_star = np.empty(self.A)
    for a in range(self.A):
      if sampling_flag:
        z_star[a]=np.random.normal(loc=self.mu[a][c_star[a]],scale=1/(self.lam[a][c_star[a]])**0.5) #sampling z
      else:
        z_star[a]=self.mu[a][c_star[a]] #argmax
    return z_star

  def logpdf(self,z,mu,lam):
    return -0.5 * (lam*(z-mu)**2-np.log(lam)+np.log(2*math.pi))

  def learn(self,w,z,mutual_iteration,model_dir,N_list,truth_category,reset_parameter=True):
    if mutual_iteration == 0 or reset_parameter:
      self.initialize_parameter(w,z,N_list)
    else:
      self.receive_z(z)
    for i in range(self.MAXITER):
      s = time.time()
      self.sampling_c()
      self.sampling_F()      
      self.sampling_T0()
      self.sampling_T()
      self.sampling_theta()
      self.sampling_pi()
      self.sampling_mu_lam()
      
      if i==0 or (i+1) % (self.MAXITER // 5) ==0 or i==self.MAXITER-1:
      #if i==0 or (i+1) % 2 ==0 or i==self.MAXITER-1:
        print(i+1,f"time:{np.round(time.time()-s,decimals=3)}",calc_ARI(self.c,truth_category))
        existing_file = glob.glob(os.path.join(model_dir,f"iter=*_mutual_iteration={mutual_iteration}.npy"))
        np.save(os.path.join(model_dir,f"iter={i+1}_mutual_iteration={mutual_iteration}"),{"z":self.z,"w":self.w,"F":self.F,"T0":self.T0,"T":self.T,"theta":self.theta,"pi":self.pi,"mu":self.mu,"lam":self.lam,"c":self.c})
        for f in existing_file:
          os.remove(f)
    
    mu_hat = np.array([self.mu[a][self.c[:,a]] for a in range(self.A)]).T
    Sigma_hat = np.array([1/self.lam[a][self.c[:,a]] for a in range(self.A)]).T
    phi_hat = [mu_hat,Sigma_hat]
    return phi_hat

if __name__ == "__main__": 
  data = np.load("save_z/beta_z_mu_beta=20_latent_dim=10_linear_dim=1024_prior_logvar=0_3dshapes_one_view_point_55544_3.npy",allow_pickle=True).item()
  label = np.load("save_z/label_3dshapes_one_view_point_55544_0.npy").astype(np.int8)
  z = data["z"]
  w = label_to_vocab(label)[:,:5]
  model = CSL_Module()
  model.learn(w,z,0)