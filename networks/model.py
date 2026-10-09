import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.net_factory import *

class model_with_dual_decoder(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, model_type = "norm"):
        if model_type == "norm": E_type = "norm"; D_type = "norm"
        if model_type == "mamba": E_type = "mamba"; D_type = "norm"
        
        self.E = ender(in_channels, out_channels,  ender_type=model_type)
        self.D = deder(in_channels, out_channels)
        self.D_ema = deder(in_channels, out_channels, ema = True)
        
    
        