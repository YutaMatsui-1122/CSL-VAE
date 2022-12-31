from utils import *

class BaseModel():
    def __init__(self,beta,file_name_option,mutual_iteration_number) -> None:
        self.mutual_iteration_number = mutual_iteration_number
        self.phi = []
        self.beta = beta
        self.latent_dim = 10
        self.linear_dim = 1024
        self.image_size = 64
        self.batch_size = 500
        self.file_name_option = file_name_option
        self.D,_,_,self.dataloader,self.shuffle_dataloader,self.w= create_dataloader(self.batch_size,self.file_name_option)