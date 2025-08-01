from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np
import os, re
from Denoising_indicator import *
from utils.matplotlib_util import *
from natsort import natsorted


class PairedCTDataset(Dataset):
    def __init__(self, data_dir, patient_ids, load_mode=0, patch_n=None, patch_size=None, MIN_HU=-1024, MAX_HU=1000):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.min_hu = MIN_HU
        self.max_hu = MAX_HU

        self.samples = []
        all_files = os.listdir(data_dir)
        input_files = natsorted([f for f in all_files if f.endswith('_input.npy')])
        
        
        for f in input_files:
            base = f.replace('_input.npy', '')
            pid = base.split('_')[0]
            target_file = f"{base}_target.npy"
            if pid in patient_ids and target_file in all_files:
                inp_path = os.path.join(data_dir, f)
                tgt_path = os.path.join(data_dir, target_file)
                input_arr = np.load(inp_path) if load_mode == 1 else inp_path
                target_arr = np.load(tgt_path) if load_mode == 1 else tgt_path

                if patch_n and patch_size:
                    if load_mode == 0:
                        input_arr = np.load(input_arr)
                        target_arr = np.load(target_arr)
                    
                    input_arr = normalize_(input_arr, self.min_hu, self.max_hu)
                    target_arr = normalize_(target_arr, self.min_hu, self.max_hu)

                    h, w = input_arr.shape
                    for _ in range(patch_n):
                        top = np.random.randint(0, h - patch_size)
                        left = np.random.randint(0, w - patch_size)
                        patch_inp = input_arr[top:top + patch_size, left:left + patch_size]
                        patch_tgt = target_arr[top:top + patch_size, left:left + patch_size]
                        self.samples.append((patch_inp, patch_tgt))
                else:
                    if load_mode == 1:
                        input_arr = normalize_(input_arr, self.min_hu, self.max_hu)
                        target_arr = normalize_(target_arr, self.min_hu, self.max_hu)
                    self.samples.append((input_arr, target_arr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_img, target_img = self.samples[idx]

        if isinstance(input_img, str):
            input_img = np.load(input_img)
            target_img = np.load(target_img)
            input_img = normalize_(input_img, self.min_hu, self.max_hu)
            target_img = normalize_(target_img, self.min_hu, self.max_hu)

        return {"input": input_img, "target": target_img}











