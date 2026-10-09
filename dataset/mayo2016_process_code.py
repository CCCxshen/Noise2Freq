import os, sys
import numpy as np
from natsort import natsorted 
import pydicom as pyi

dataset_name = "mayo2016_3mm"

dataset_path = "C:\\Users\\Sxc\\Desktop\\123\\mayo2016_3mm"
save_path = "./mayo2016_3mm_processed"

os.makedirs(save_path, exist_ok=True)

IDs = os.listdir(dataset_path)

def read_dicoms(slices_list, path):
    slices = []
    
    for slice_name in slices_list:
        slices.append(pyi.dcmread(os.path.join(path, slice_name)))
        
    slices = natsorted(slices, key = lambda x: x.SliceLocation)
    return slices


for idx_ID, ID in enumerate(IDs):
    CTs = natsorted(os.listdir(os.path.join(os.path.join(dataset_path, ID))))
    NDCTs = natsorted(os.listdir(os.path.join(dataset_path, ID, CTs[0])))
    LDCTs = natsorted(os.listdir(os.path.join(dataset_path, ID, CTs[1])))
    
    NDCT_slices = read_dicoms(NDCTs, os.path.join(dataset_path, ID, CTs[0]))
    LDCT_slices = read_dicoms(LDCTs, os.path.join(dataset_path, ID, CTs[1]))
    
    ID_txt = f"[{ID}: {idx_ID + 1} / {len(IDs)}]"
    for i in range(len(NDCT_slices)):
        NDCT_array = NDCT_slices[i].pixel_array * NDCT_slices[i].RescaleSlope + NDCT_slices[i].RescaleIntercept
        LDCT_array = LDCT_slices[i].pixel_array * LDCT_slices[i].RescaleSlope + LDCT_slices[i].RescaleIntercept
        
        NDCT_array = np.clip(NDCT_array, -1024, 1000)
        LDCT_array = np.clip(LDCT_array, -1024, 1000)
        
        np.save(os.path.join(save_path, f"{ID}_{i+1}_target"), NDCT_array)
        np.save(os.path.join(save_path, f"{ID}_{i+1}_input"), LDCT_array)
        
        slice_txt = f"[slices: {i+1} / {len(NDCT_slices)}]"
        print(ID_txt+slice_txt, end="\n")
    