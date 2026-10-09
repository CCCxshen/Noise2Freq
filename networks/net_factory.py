from networks.en_and_de import *
from networks.FusionModule import *
from networks.edge_network import *
from networks.Uformer import Uformer_encoder, Uformer_decoder
import torch.nn as nn

def ender(in_chns=1, class_num=1, ema=False, encoder_type = "f"):
    if encoder_type == "norm": net = encoder(in_chns)
    if encoder_type == "MCCM": net = mamba_encoder_parallel(in_chns)
    if encoder_type == "Uformer": 
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        net = Uformer_encoder(img_size=256, embed_dim=16, depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def deder(in_chns=1, class_num=1, ema=False, decoder_type = "norm"):
    if decoder_type == "norm" or decoder_type == "MCCM": net = decoder(in_chns, class_num)
    if decoder_type == "Uformer": 
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        net = Uformer_decoder(img_size=256, embed_dim=16, depths=depths,
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def fusion(alpha = 0, ema=False):
    net = FusionModule(alpha = alpha)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def edge(alpha = 0, ema=False):
    net = CTHighFreqExtractor()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net