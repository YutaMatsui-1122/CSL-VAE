import torch,os,glob,time,math,cv2,copy
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import save_model,HSV2RGB,create_dataloader
from utils import label_to_word
from torchinfo import summary

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) # [,]
class Image_Classification_Net(nn.Module):
    def __init__(self,V):
        super(Image_Classification_Net, self).__init__()
        layer = 4
        image_channel=3
        image_size = 64
        channels=[image_channel,32,64,128,256]
        kernel = np.repeat(4,layer)
        stride = np.repeat(2,layer)
        padding = np.repeat(1,layer)
        modules=[]
        for h in range(len(channels)-1):
            modules.append(
                nn.Conv2d(in_channels=channels[h], out_channels=channels[h+1], kernel_size=kernel[h], stride=stride[h], padding=padding[h])
            )
            modules.append(
                nn.ReLU()
            )
        Linear1dim = int((image_size/(2**layer))*(image_size/(2**layer))*channels[-1])
        Linear2dim = Linear1dim//2
        Linear3dim = Linear2dim//2
        Linear4dim = Linear3dim//2
        modules.append(Flatten())
        modules.append(nn.Linear(Linear1dim, Linear2dim))
        modules.append(nn.Linear(Linear2dim, Linear3dim))
        modules.append(nn.Linear(Linear3dim, Linear4dim))
        modules.append(nn.Linear(Linear4dim, V))
        modules.append(nn.Sigmoid())
        self.net = nn.Sequential(*modules)

    def forward(self,image):
        word_sequence = self.net(image)
        return word_sequence
    
    def loss_function(self,label,truth_label):
        BCE = F.binary_cross_entropy(label, truth_label, reduction='sum')
        return BCE
    
    def learn(self,shuffle_dataloader,epoch,file_name_option):
        model=copy.deepcopy(self.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)        
        for i in range(epoch):
            train_loss = 0
            s = time.time()
            D = 0
            for image,truth_label,_ in shuffle_dataloader:
                image,truth_label=image.to(device),truth_label.to(device).float()
                optimizer.zero_grad()
                label = model.forward(image)
                loss = self.loss_function(label,truth_label)
                loss.backward()
                train_loss += loss.item()
                D += image.shape[0]
                optimizer.step()
            if i==0 or (i+1) %  (epoch // 10) == 0 or i == (epoch-1):
                print(f'====> Epoch: {i+1} Average loss: {train_loss / D}  Learning time:{np.round(time.time()-s,decimals=3)}')
                self.save_model(model,file_name_option,i)
    
    def save_model(self,model,file_name_option,iter):
        save_dir = f"exp_CSL_VAE/ICN"
        existfiles = glob.glob(os.path.join(save_dir,f"{file_name_option}_epoch=*.pth"))
        os.makedirs(save_dir,exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir,f"{file_name_option}_epoch={iter+1}.pth"))
        for f in existfiles:
            os.remove(f)
    
    def setting_learnt_model(self,file_name_option):
        save_dir = f"exp_CSL_VAE/ICN"
        model_file = glob.glob(os.path.join(save_dir,f"{file_name_option}_epoch=*.pth"))[0]
        self.load_state_dict(torch.load(model_file))
    
    def predict(self,image):
        image.to(device)
        label = self(image).to("cpu").round().to(torch.int)
        return label
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
file_name_option = None
dataset_name = "3dshapes"
file_name_option = "six_view_point_66644_2"
exp = 0
exp_dir = f"exp_CSL_VAE/exp{exp}"
word_sequence_setting_file = os.path.join(exp_dir,"word sequence setting.npy")
data = np.load(word_sequence_setting_file,allow_pickle=True).item()
valid_list,truth_F,truth_T0,truth_T = data.values()
_,_,_,w,_,full_shuffle_loader,_,_,_= create_dataloader(batch_size,file_name_option,valid_list,khot_flag=True)

if __name__ == "__main__":
    icn = Image_Classification_Net(V=w.shape[1])
    icn.learn(full_shuffle_loader,1000,file_name_option=file_name_option)
    icn.setting_learnt_model(file_name_option=file_name_option)
    truth_counter = 0
    D = 0
    for image,truth_label,_ in full_shuffle_loader:
        image = image.to(device)
        predicted_labels = icn.predict(image).to(torch.int8)
        TF_array = [torch.allclose(predicted_labels[i],truth_label[i]) for i in range(image.shape[0])]
        truth_counter += torch.count_nonzero(torch.tensor(TF_array))
        D+=len(TF_array)
    
    print(D,truth_counter/D)
