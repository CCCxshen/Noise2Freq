import os, sys
import numpy as np
from natsort import natsorted 
import pydicom as pyi

dataset_name = "mayo2020_5mm"

dataset_path = "C:\\Users\\Sxc\\Desktop\\123\\mayo2020_5mm"
save_path = "./mayo2020_5mm_processed"

os.makedirs(save_path, exist_ok=True)

IDs = os.listdir(dataset_path)

def read_dicoms(slices_list_1, slices_list_2, path, CTs):
    slices_1 = []
    slices_2 = []
    
    for slice_name in slices_list_1:
        slices_1.append(pyi.dcmread(os.path.join(path, CTs[0], slice_name)))
        
    for slice_name in slices_list_2:
        slices_2.append(pyi.dcmread(os.path.join(path, CTs[1], slice_name)))
    
    print(slices_1[0].SliceLocation)
    slices_1 = natsorted(slices_1, key = lambda x: x.SliceLocation)
    slices_2 = natsorted(slices_2, key = lambda x: x.SliceLocation)
    print(slices_1[0].SliceLocation)
    
    
    if slices_1[0].SeriesDescription == "Full Dose Images":
        return slices_1, slices_2
    else: return slices_2, slices_1
    


for idx_ID, ID in enumerate(IDs):
    CTs = natsorted(os.listdir(os.path.join(os.path.join(dataset_path, ID))))
    CTs_1 = natsorted(os.listdir(os.path.join(dataset_path, ID, CTs[0])))
    CTs_2 = natsorted(os.listdir(os.path.join(dataset_path, ID, CTs[1])))
    
    NDCT_slices, LDCT_slices = read_dicoms(CTs_1, CTs_2, os.path.join(dataset_path, ID), CTs)
    
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
    
    