# Requirements
import os
import numpy as np
import SimpleITK as sitk
import h5py
import cv2

# Path to the training folder
train_path = "data/training/"

def mhd_to_array(path):
    """
    Read a *.mhd file stored in path and return it as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))

# Define lists storing all images names

patients_list = os.listdir(train_path)


# train_2ch_frames_list = sorted(os.listdir(train_path + "2ch/frames/"))
# train_2ch_masks_list = sorted(os.listdir(train_path + "2ch/masks/"))
# train_4ch_frames_list = sorted(os.listdir(train_path + "4ch/frames/"))
# train_4ch_masks_list = sorted(os.listdir(train_path + "4ch/masks/"))

# Create hierarchical h5py file, ready to be filled with 4 datasets
f = h5py.File("data/image_dataset.hdf5", "w")

f.create_dataset("train 2ch frames", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "float32")

f.create_dataset("train 2ch masks", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "int32")

f.create_dataset("train 4ch frames", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "float32")

f.create_dataset("train 4ch masks", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "int32")

counter_dict = {"train 2ch frames": 0,
                "train 2ch masks": 0,
                "train 4ch frames": 0,
                "train 4ch masks": 0}



for patient in patients_list:
    files_list = os.listdir(os.path.join(train_path, patient))
    for f_name in files_list:
        if "mhd" in f_name and not "Zone" in f_name and not 'sequence' in f_name:
            array = mhd_to_array(os.path.join(train_path, patient, f_name))
            if "gt" in f_name:
                new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_NEAREST)
                name_end = 'masks'
            elif "ES" in f_name or "ED" in f_name:
                new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
                new_array = new_array/255
                name_end = 'frames'

            file_list = f_name.split('_')
            new_array = np.reshape(new_array,(384,384,1))
            name = f'train {file_list[1].lower()} {name_end}'
            f[name][counter_dict[name],...] = new_array[...]
            counter_dict[name] += 1


# Iterate over images and save them into the corresponding dataset
# j = 0
# for i in train_2ch_frames_list:
#     if "mhd" in i:
#         array = mhd_to_array(os.path.join(train_path, "2ch/frames", i))
#         new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
#         new_array = np.reshape(new_array,(384,384,1))
#         new_array = new_array/255
#         f["train 2ch frames"][j,...] = new_array[...]
#         j = j + 1        
        
# j = 0
# for i in train_2ch_masks_list:
#     if "mhd" in i and 'Zone' not in i:
#         array = mhd_to_array(os.path.join(train_path, "2ch/masks", i))
#         new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_NEAREST)
#         new_array = np.reshape(new_array,(384,384,1))
#         f["train 2ch masks"][j,...] = new_array[...]
#         j = j + 1   
        
# j = 0
# for i in train_4ch_frames_list:
#     if "mhd" in i:
#         array = mhd_to_array(os.path.join(train_path, "4ch/frames", i))
#         new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
#         new_array = np.reshape(new_array,(384,384,1))
#         new_array = new_array/255
#         f["train 4ch frames"][j,...] = new_array[...]
#         j = j + 1        
        
# j = 0
# for i in train_4ch_masks_list:
#     if "mhd" in i:
#         array = mhd_to_array(os.path.join(train_path, "4ch/masks", i))
#         new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_NEAREST)
#         new_array = np.reshape(new_array,(384,384,1))
#         f["train 4ch masks"][j,...] = new_array[...]
#         j = j + 1       
        
f.close()

