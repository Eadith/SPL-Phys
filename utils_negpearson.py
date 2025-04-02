from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn


class Neg_Pearson(nn.Module):
    """
    The Neg_Pearson Module is from the orignal author of Physnet.
    Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
    source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
    """
    
    def __init__(self, device):
        super(Neg_Pearson, self).__init__()
        self.device = device

    def forward(self, preds, labels):       
        loss = torch.zeros(preds.shape[0], device=self.device)

        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss[i] = (1 - pearson)

        return loss
    
if __name__ == '__main__':
    y = torch.randn(4,160).cuda()
    y1 = torch.rand(4,160).cuda()
    macc = Neg_Pearson()
    loss = macc(y,y)
    print(loss)