import numpy  as np
file = "exp_CSL_VAE/exp2/word sequence setting.npy"
data = np.load(file,allow_pickle=True).item()
a,b,c = data.values()
print(a)