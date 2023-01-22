from CSL_VAE import CSL_VAE
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import adjusted_rand_score as ARI
import torch, glob,time
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from image_classification_net import Image_Classification_Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


exp = 2

batch_size = 100
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "six_view_point_66644_2"
Attribute_num = 5
word = label_to_word(file_name_option)
label = np.load(f"dataset/{dataset_name}_hsv_labels_{file_name_option}.npy")
w=label_to_vocab(label[:,:5])

exp_dir = f"exp_CSL_VAE/exp{exp}"
model_dir = os.path.join(exp_dir,"model")
result_dir = os.path.join(exp_dir,"result")
os.makedirs(result_dir,exist_ok=True)

word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
setting = np.load(word_sequence_setting_file,allow_pickle=True).item()
valid_list,truth_F,truth_T0,truth_T = setting.values()
CSL_Module_parameters,VAE_Module_parameters = np.load(f"exp_CSL_VAE/exp{4}/Hyperparameters.npy",allow_pickle=True).item().values()

dataloader,valid_loader,shuffle_dataloader,w,_,full_shuffle_loader,full_dataset,_,_= create_dataloader(batch_size,file_name_option,valid_list)
w = w.to('cpu').detach().numpy().copy()
truth_category = np.copy(w)
new_data = False
D=full_dataset.image.shape[0]

for unknown_data,unknown_label,_ in valid_loader :
    break

for known_data,known_label,_ in shuffle_dataloader:
    break
I = 4

known_label = known_label.numpy()[:,:5]

unknown_label = unknown_label.numpy()[:,:5]
N_star = 5

def calc_ARI_EAR(label,truth_labels,F,truth_F):
    ARI_list = []
    ARI_index = []
    for a in range(truth_labels.shape[1]):        
        ARI_list.append(np.max([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
        ARI_index.append(np.argmax([ARI(label[:,i],truth_labels[:,a]) for i in range(label.shape[1])]))
    D,N_max = truth_F.shape
    A = np.max(F)
    N = np.array([np.count_nonzero(truth_F[d]<5) for d in range(D)])
    truth_F = np.array([[np.array(ARI_index)[truth_F[d][n]] if n<N[d] else A  for n in range(N_max)] for d in range(D)])
    print(np.sum([np.count_nonzero(truth_F[d][:N[d]]==F[d][:N[d]])for d in range(D)])/np.sum(N))
    return np.round(ARI_list,decimals=3), ARI_index

def tensor_equal(a,b):
    if len(a)!=len(b):
        return False
    elif torch.allclose(a,b):
        return True
    else:
        return False

def multi_label_metrics(T_lengths, Y_lengths, Y_and_T_lengths, Y_or_T_lengths):
    #refer to https://qiita.com/jyori112/items/110596b4f04e4e1a3c9b
    #         https://www.researchgate.net/publication/266888594_A_Literature_Survey_on_Algorithms_for_Multi-label_Learning
    n = len(T_lengths)
    accuracy  = 1 / n * np.sum(Y_and_T_lengths / Y_or_T_lengths)
    recall    = 1 / n * np.sum(Y_and_T_lengths / T_lengths)
    precision = 1 / n * np.sum(Y_and_T_lengths / Y_lengths)
    F_measure = 2 * (precision * recall) / (precision + recall)
    return accuracy, recall, precision,F_measure

def wrd2img_plt():
    for iter in np.arange(20):        
        result_MI_dir = os.path.join(result_dir,f"MI{iter}")
        os.makedirs(result_MI_dir,exist_ok=True)        
        csl_vae = CSL_VAE(file_name_option,mutual_iteration_number=11,CSL_Module_parameters=CSL_Module_parameters,VAE_Module_parameters=VAE_Module_parameters,valid_list=valid_list)
        csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir) 
        fig, axes = plt.subplots(2,I, figsize=((I,2)))
        for i in range(I):
            w_star = unknown_label[i*6][:5]
            o_star = csl_vae.wrd2img(w_star)
            word_sequence = ",  ".join(word[unknown_label[i*10]])
            fig.suptitle(f"wrd2img inference (ML {iter})")
            axes[0][i].tick_params(bottom=False, left=False, right=False, top=False);axes[1][i].tick_params(bottom=False, left=False, right=False, top=False)
            axes[0][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False);axes[1][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            axes[0][i].imshow(HSV2RGB(unknown_data[i*10]))
            #axes1.set_title("correct image")
            axes[1][i].imshow(HSV2RGB(o_star))
            #axes2.set_title("infered image by CSL+VAE")
        plt.savefig(os.path.join(result_MI_dir,f"Wrd2img_unknown"))
        fig, axes = plt.subplots(2,I, figsize=((I,2)))
        for i in range(I):
            w_star = known_label[i][:5]
            o_star = csl_vae.wrd2img(w_star)
            word_sequence = ",  ".join(word[known_label[i]])
            fig.suptitle(f"wrd2img inference (ML {iter})")
            axes[0][i].tick_params(bottom=False, left=False, right=False, top=False);axes[1][i].tick_params(bottom=False, left=False, right=False, top=False)
            axes[0][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False);axes[1][i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)

            axes[0][i].imshow(HSV2RGB(known_data[i]))
            #axes1.set_title("correct image")
            axes[1][i].imshow(HSV2RGB(o_star))
            #axes2.set_title("infered image by CSL+VAE")
        plt.savefig(os.path.join(result_MI_dir,f"Wrd2img_known"))
        I_oneword=5
        fig, axes = plt.subplots(1,I_oneword, figsize=((I_oneword,1.5)))
        for words in [[0],[1,8],[4,9,14],[2,11,16,24]]:
            for i in range(I_oneword):
                w_star = np.array(words)
                o_star = csl_vae.wrd2img(w_star)
                word_sequence = ",  ".join(word[w_star])
                fig.suptitle(f"\"{word_sequence}\"")
                axes[i].tick_params(bottom=False, left=False, right=False, top=False)
                axes[i].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
                axes[i].imshow(HSV2RGB(o_star))
            plt.savefig(os.path.join(result_MI_dir,f"Wrd2img_{len(w_star)}_word"))

        
def img2wrd():
    for iter in np.arange(20):
        existing_file = glob.glob(os.path.join(model_dir,f"CSL_Module/iter=*_mutual_iteration={iter}.npy"))[0]
        data = np.load(existing_file,allow_pickle=True).item()
        c=data["c"]
        F=data["F"]
        _,F_index=calc_ARI_EAR(c,truth_category,F,truth_F)   
        result_MI_dir = os.path.join(result_dir,f"MI{iter}")
        os.makedirs(result_MI_dir,exist_ok=True)
        csl_vae = CSL_VAE(file_name_option,mutual_iteration_number=11,CSL_Module_parameters=CSL_Module_parameters,VAE_Module_parameters=VAE_Module_parameters,valid_list=valid_list)
        csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir)
        csl_vae.vae_module.to(device)
        for i in range(2):
            Truth_word_sequence = ",  ".join(word[known_label[i]])
            xlabel = f"{Truth_word_sequence}"
            for j in range(10):
                o_star = known_data[i]
                w_star = csl_vae.img2wrd(o_star,sampling_method=5)
                word_sequence=",  ".join(word[w_star])
                xlabel+=f"\n\n{word_sequence}"
            plt.title(f"Img2wrd (ML {iter})",fontsize=18)
            plt.xlabel(xlabel,fontsize=18)
            plt.imshow(HSV2RGB(known_data[i]))
            plt.tick_params(bottom=False, left=False, right=False, top=False)
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            print(os.path.join(result_MI_dir,f"Img2wrd_{i}"))
            plt.savefig(os.path.join(result_MI_dir,f"Img2wrd_{i}"))

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

def wrd2img_quant_eval():
    f = open(os.path.join(result_dir,'Wrd2img evaluation by image classifier.txt'), 'w', newline='\n')
    icn = Image_Classification_Net(V=np.max(w)+1)
    icn.setting_learnt_model(file_name_option)
    for iter in range(10):
        result_MI_dir = os.path.join(result_dir,f"MI{iter}")
        os.makedirs(result_MI_dir,exist_ok=True)
        csl_vae = CSL_VAE(file_name_option,mutual_iteration_number=11,CSL_Module_parameters=CSL_Module_parameters,VAE_Module_parameters=VAE_Module_parameters,valid_list=valid_list)
        csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir) 
        truth_counter = 0
        for _,label,_ in full_shuffle_loader:
            temp_batch_size = label.shape[0]
            w_stars = label[:,:5]
            o_stars = torch.zeros((temp_batch_size,3,64,64))
            for i in range(temp_batch_size):
                o_stars[i] = csl_vae.wrd2img(w_stars[i])
            icn_w_stars_knot = icn.predict(o_stars)
            icn_w_stars = [torch.where(icn_w_stars_knot[i])[0].to(torch.int8) for i in range(temp_batch_size)]
            TF_array = torch.Tensor([tensor_equal(icn_w_stars[i],w_stars[i]) for i in range(temp_batch_size)])
            truth_counter += torch.count_nonzero(TF_array)       
        f.write(f"MI {iter}\n")
        f.write(f"  Wrd2img accuracy = {truth_counter / D}\n\n")
        print(f"MI {iter} : Wrd2img accuracy = {truth_counter / D}")
    f.close()
        

def img2wrd_quant_eval():
    f = open(os.path.join(result_dir,'Img2wrd metrics.txt'), 'w', newline='\n')
    for iter in [0,10]:
        existing_file = glob.glob(os.path.join(model_dir,f"CSL_Module/iter=*_mutual_iteration={iter}.npy"))[0]
        data = np.load(existing_file,allow_pickle=True).item()
        result_MI_dir = os.path.join(result_dir,f"MI{iter}")
        os.makedirs(result_MI_dir,exist_ok=True)
        csl_vae = CSL_VAE(file_name_option,mutual_iteration_number=11,CSL_Module_parameters=CSL_Module_parameters,VAE_Module_parameters=VAE_Module_parameters,valid_list=valid_list)
        csl_vae.setting_learned_model(w,beta=16,file_name_option=file_name_option,mutual_iteration=iter,model_dir=model_dir)
        csl_vae.vae_module.to(device)
        
        w_star_lengths = np.empty(D)
        w_truth_lengths = np.full(D,Attribute_num)
        star_and_truth = np.empty(D)
        star_or_truth = np.empty(D)

        d=0
        f.write(f"MI {iter}\n")
        for loader_name,loader in zip(["valid","train"],[valid_loader,shuffle_dataloader]):
            f.write(f"  {loader_name}\n")
            for image,label,_ in loader:
                temp_batch_size = image.shape[0]
                image=image.to(device)
                label=label[:,:5]
                for i in range(temp_batch_size):
                    w_star = csl_vae.img2wrd(image[i],sampling_method=5)
                    w_star_lengths[d] = len(w_star)
                    star_and_truth[d] = len(np.intersect1d(label[i].numpy(),w_star))
                    star_or_truth[d]  = len(np.union1d(label[i].numpy(),w_star))
                    d+=1
            if loader_name == "valid":
                D_valid = d
                accuracy, recall, precision,F_measure = multi_label_metrics(w_truth_lengths[:D_valid],w_star_lengths[:D_valid],star_and_truth[:D_valid],star_or_truth[:D_valid])
                f.write(f"      D : {D_valid}, accuracy : {accuracy}, recall : {recall}, precision : {precision}, F1 : {F_measure}\n\n")
            if loader_name == "train":
                accuracy, recall, precision,F_measure = multi_label_metrics(w_truth_lengths[D_valid:d],w_star_lengths[D_valid:d],star_and_truth[D_valid:d],star_or_truth[D_valid:d])
                f.write(f"      accuracy : {accuracy}, recall : {recall}, precision : {precision}, F1 : {F_measure}\n\n")
        f.write(f"  Full\n")
        accuracy, recall, precision,F_measure = multi_label_metrics(w_truth_lengths[:d],w_star_lengths[:d],star_and_truth[:d],star_or_truth[:d])
        f.write(f"      accuracy : {accuracy}, recall : {recall}, precision : {precision}, F1 : {F_measure}\n\n")
    f.close()


#img2wrd()
img2wrd_quant_eval()
#wrd2img_plt()
#wrd2img_quant_eval()