from networks.en_and_de import *
from networks.FusionModule import *
import torch.nn as nn

def ender(in_chns=1, class_num=1, ema=False):
    net = encoder(in_chns)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def deder(in_chns=1, class_num=1, ema=False):
    net = decoder(in_chns, class_num)
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