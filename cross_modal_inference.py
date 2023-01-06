from CSL_VAE import CSL_VAE
from VAE_Module import VAE_Module
from CSL_Module_ak import CSL_Module

from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch, glob
from utils import *
import numpy as np
import matplotlib.pyplot as plt


batch_size = 500
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "five_view_point_55544"
dup_num = 2
shift_rate = 0.01

Attribute_num = 5

word = label_to_word(file_name_option)
file_name_option += f"×{dup_num}_shift_{shift_rate}"
label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
w=label_to_vocab(label[:,:5])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csl_vae = CSL_VAE()
exp = 12
valid_list = [[3,6,11,16,20],[0,5,14,17,22],[4,7,12,16,19],[1,9,10,18,21]]
exp_dir = f"exp_CSL_VAE/exp{exp}"
model_dir = os.path.join(exp_dir,"model")
result_dir = os.path.join(exp_dir,"result")
os.makedirs(result_dir,exist_ok=True)
word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
data = np.load(word_sequence_setting_file,allow_pickle=True).item()
valid_list,grammar_list,Nd_rate = data.values()
dataloader,valid_loader,shuffle_dataloader,w= create_dataloader(batch_size,file_name_option,valid_list)

for data,label,_ in valid_loader:
    break
w = w.to('cpu').detach().numpy().copy()

label = label.numpy()[:,:5]

N_star = 5

I =15

for iter in range(9):
    fig, axes = plt.subplots(2,I, figsize=((I,2)))
    csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir,valid_list=valid_list,grammar_list=grammar_list,Nd_rate=Nd_rate) 
    for i in range(I):
        i1=1
        w_star = label[i1][:5]
        o_star = csl_vae.wrd2img(w_star)
        word_sequence = ",  ".join(word[label[i1]])
        fig.suptitle(f"wrd2img inference (ML {iter})")
        axes[0][i].tick_params(bottom=False, left=False, right=False, top=False);axes[1][i].tick_params(bottom=False, left=False, right=False, top=False)
        axes[0][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False);axes[1][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

        axes[0][i].imshow(HSV2RGB(data[i1]))
        #axes1.set_title("correct image")
        axes[1][i].imshow(HSV2RGB(o_star))
        #axes2.set_title("infered image by CSL+VAE")
    plt.savefig(os.path.join(result_dir,f"Wrd2img_ML {iter}"))

    for i in range(2):
        figure = plt.figure(figsize=(10,10))
        o_star = data[i]
        w_star = csl_vae.img2wrd(o_star,N_star)
        word_sequence = ",  ".join(word[w_star])
        Truth_word_sequence = ",  ".join(word[label[i]])
        print("infered word sequence \"",word_sequence,"\"")
        plt.title(f"Img2wrd (ML {iter})",fontsize=18)
        plt.xlabel(f"{Truth_word_sequence}\n\n{word_sequence}" ,fontsize=18)
        plt.imshow(HSV2RGB(data[i]))
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.savefig(os.path.join(result_dir,f"Img2wrd_ML {iter}_{i}"))

    '''word_sequence_accuracy = []
    word_accuracy = []
    for data,label,_ in csl_vae.vae_module.shuffle_dataloader:
        for i in range(len(data)):
            o_star = data[i]
            w_star = csl_vae.img2wrd(o_star,N_star)
            predicted_word = word[w_star]
            truth_word = word[label[i][:5]]
            word_accuracy.append(np.count_nonzero(predicted_word==truth_word))
            word_sequence_accuracy.append(np.count_nonzero(predicted_word==truth_word)==5)'''
    
    """print(f"ML {iter}",np.mean(word_accuracy))
    print(f"ML {iter}",np.count_nonzero(word_sequence_accuracy)/ len(word_sequence_accuracy))"""

