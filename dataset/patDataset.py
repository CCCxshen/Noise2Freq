from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np
import os, re
from Denoising_indicator import *

class patDataset(Dataset):
    def __init__(self, data_dir, is_train, regex_pattern = [r'.*'], process_fun = None, args = None):
        self.data_dir = os.path.abspath(data_dir)
        self.is_train = is_train
        self.process_fun = process_fun
        self.args = args
        
        regex = [re.compile(x) for x in regex_pattern]
        self.data_names = [natsorted([x for x in os.listdir(self.data_dir) if y.match(x)]) for y in regex]
        self.data_paths = np.array([[os.path.join(self.data_dir, y) for y in x] for x in self.data_names])
    
    def __len__(self):
        return self.data_paths.shape[1]
    
    def __getitem__(self, index):
        data_path = np.array([x[index] for x in self.data_paths])

        # if self.process_fun != None:
        #     data = self.process_fun(data_path, self.is_train, self.args)
        I = normalize_(np.load(data_path[0]), -1024, 1000)
        T = normalize_(np.load(data_path[1]), -1024, 1000)
        # T = I

        return {
            "input": I,
            "target": T
        }