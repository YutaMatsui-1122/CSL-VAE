import cv2
from cv2 import COLOR_RGB2HSV
from cv2 import COLOR_HSV2RGB
from cv2 import imshow
import numpy as np
from matplotlib import pyplot as plt
import h5py
from torch import select_copy
from file_option_name_memo import file_name_option_dict
from keras.preprocessing.image import ImageDataGenerator
import gc

def eliminate_element(eliminate_list,image,label):
    original_list = np.arange(label.shape[0])
    full_eliminate_list = []
    for factor in range(6):
        for index in eliminate_list[factor]:
            full_eliminate_list = np.append(full_eliminate_list,np.where(label[:,factor]==index)[0])
    full_eliminate_list = np.unique(full_eliminate_list).astype(np.int32)
    selected_list = np.delete(original_list,full_eliminate_list)
    image = image[selected_list]
    print("finish image")
    label = label[selected_list]
    print("finish label")
    return image, label

def label_to_int(label):
    int_label = np.empty_like(label,dtype=np.int8)
    for i in range(6):
        label_list = np.unique(label[:,i])
        for j, label_num in enumerate(label_list):
            index = np.where(label[:,i]==label_num)[0]
            int_label[:,i][index] = j        
    return int_label

def duplication(generator,images,dup_num):
    duplicated_images = -np.ones((len(images)*dup_num,*images[0].shape),dtype=np.float32)
    for index,image in enumerate(images):
        dup_num_image = np.repeat(image.reshape((1, *image.shape)), dup_num, axis=0)
        gen = generator.flow(dup_num_image/255, batch_size = dup_num) ## divide 255 for the "ImageDataGenerator"
        duplicated_images[index*dup_num : (index+1)*dup_num] = next(gen) * 255
    return duplicated_images


#load dataset 480000 × 64 × 64 × 3
# dataset detail →　https://github.com/deepmind/3d-shapes

f = h5py.File('dataset/3dshapes.h5', 'r')

dataset = f["images"]
label = f["labels"]
num_file=dataset.shape[0]
del f
gc.collect()
label = label_to_int(label)

file_name_option = "six_view_point_66644_2"
eliminate_list = file_name_option_dict[file_name_option]

original_list = np.arange(label.shape[0])
full_eliminate_list = []
for factor in range(6):
    for index in eliminate_list[factor]:
        full_eliminate_list = np.append(full_eliminate_list,np.where(label[:,factor]==index)[0])

full_eliminate_list = np.unique(full_eliminate_list).astype(np.int32)
selected_list = np.delete(original_list,full_eliminate_list)
label = label[selected_list]

sep_selected_list = np.split(selected_list,16)
sep_dataset = []
i = 0

dup_num = 1
shift_rate = 0
width_datagen = ImageDataGenerator(height_shift_range = shift_rate ,width_shift_range=shift_rate)

for selected_list in sep_selected_list:
    sep_dataset.append(duplication(width_datagen,dataset[selected_list],dup_num).astype(np.uint8))
    print(sep_dataset[i].max(),sep_dataset[i].min())
    print(i)
    i+=1

del dataset
gc.collect()

image_width = 64
image_height = 64
image_channel = 3
num_file = len(label)*dup_num

H_max = 255
S_max = 255
V_max = 255

save_data =np.empty((num_file,image_width,image_height,image_channel),dtype=np.float32)
i=0
print(sep_dataset[0].max(),sep_dataset[0].min())
for d in sep_dataset:
    for img in d:
        HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL).astype(np.float32)
        HSV_img[:,:,0] = HSV_img[:,:,0]/H_max
        HSV_img[:,:,1] = HSV_img[:,:,1]/S_max
        HSV_img[:,:,2] = HSV_img[:,:,2]/V_max
        save_data[i] = HSV_img
        i+=1
    print(i)


if dup_num > 1:
    file_name_option += f"×{dup_num}_shift_{shift_rate}"
label = np.repeat(label,dup_num,axis=0)
print(save_data.max(),save_data.min())
print(len(save_data))
np.save(f'dataset/3dshapes_hsv_images_{file_name_option}',save_data)
np.save(f'dataset/3dshapes_hsv_labels_{file_name_option}',label)