import numpy as np
from utils import *
from file_option_name_memo import *
    

def create_word_sequence(file_name_option,valid_list,grammar):
    _,_,_,w,_,_,_,_,_= create_dataloader(500, file_name_option, valid_list)
    truth_T0, truth_T = grammar_list[grammar].values()
    D,N_max = w.to('cpu').detach().numpy().copy().shape
    truth_F = np.zeros_like(w,dtype=np.int8)
    N = np.zeros(D,dtype=np.int8)
    total_w_num = 0
    for d in range(D):
        truth_F[d][0] = np.random.choice(N_max,p=truth_T0)
        w[d][0] = w[d][truth_F[d][0]]
        N[d] += 1
        total_w_num += 1
        for n in range(1,N_max):
            truth_F[d][n] = np.random.choice(N_max+1,p=truth_T[truth_F[d][n-1]])
            if truth_F[d][n] == N_max:
                w[d][n] = -1
            else:
                w[d][n] = w[d][truth_F[d][n]]
                N[d] += 1
                total_w_num += 1
    return D, N, w, truth_F