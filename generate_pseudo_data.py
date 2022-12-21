import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

D = 10000
K = 10
A = 10
N = 5
L = K * A
V = L // 2
attribute_num = 5

alpha_phi = 100
alpha_theta = 100
alpha_psi = 1
alpha_pi = 100

m = 0
tau = 1
a_lam = 1
b_lam = 1

truth_pi = np.random.dirichlet(np.repeat(alpha_pi,K),size=A)
truth_mu = np.vstack([np.tile(np.linspace(-2,2,K),(attribute_num,1)),np.zeros((A-attribute_num,K))])
truth_lam = np.vstack([np.full((attribute_num,K),200),np.ones((A-attribute_num,K))])
truth_theta = np.random.dirichlet(np.repeat(alpha_theta,V),size=L)
truth_theta = np.vstack([np.identity(V),np.random.dirichlet(np.repeat(alpha_theta,V),size=L-V)])
truth_psi = np.array([0.5,0,0.5,0,0,0,0,0,0,0])
truth_phi = np.vstack([np.identity(A)[[1,2,3,4,0]],np.random.dirichlet(np.repeat(alpha_phi,A),size = A-attribute_num)])

c = np.zeros((D,A),dtype=np.int8)
z = np.zeros((D,A))
F = np.zeros((D,N),dtype=np.int8)
w = np.zeros((D,N),dtype=np.int8)
for d in range(D):
  for a in range(A):
    c[d][a] = np.random.choice(K,p=truth_pi[a])
    z[d][a] = np.random.normal(loc = truth_mu[a][c[d][a]], scale=1/(truth_lam[a][c[d][a]]**0.5))
  F[d][0] = np.random.choice(A,p=truth_psi)
  w[d][0] = np.random.choice(V,p=truth_theta[F[d][0]*A+c[d][F[d][0]]])
  for n in range(1,N):
    F[d][n] = np.random.choice(A,p=truth_phi[F[d][n-1]])
    w[d][n] = np.random.choice(V,p=truth_theta[F[d][n]*A+c[d][F[d][n]]])

print(truth_theta)
print(truth_phi)
np.save(f"pseudo_data/pseudo_K={K}_A={A}_N={N}",{"z":z,"w":w,"truth_pi":truth_pi,"truth_mu":truth_mu,"truth_lam":truth_lam,"truth_theta":truth_theta,"truth_psi":truth_psi,"truth_phi":truth_phi,"truth_c":c,"truth_F":F})