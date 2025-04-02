import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft
import math
from scipy.interpolate import Akima1DInterpolator
from utils_post_process import calculate_metric_per_video


class FAL(nn.Module):
    def __init__(self, Fs, high_pass=40, low_pass=240):
        super(FAL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()

    def forward(self, pos_rppg1,pos_rppg2):
        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)

        loss = self.distance_func(posfre1, posfre2)

        return loss

class FCL(nn.Module):
    def __init__(self, Fs, high_pass=2.5, low_pass=0.4,tau=0.08):
        super(FCL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()
        self.tau=tau

    def forward(self, neg_rppgarr,pos_rppg1,pos_rppg2):

        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)
        pos_dis=torch.exp(self.distance_func(posfre1, posfre2)/self.tau)
        neg_dis_total=0
        # neg_rppgarr K,B,T
        for i in range(len(neg_rppgarr)):
            negfre=self.norm_psd(neg_rppgarr[i])
            neg_dis = torch.exp(self.distance_func(posfre1, negfre) / self.tau)+torch.exp(self.distance_func(posfre2, negfre) / self.tau)
            neg_dis_total+=neg_dis

        loss = torch.log10(pos_dis/neg_dis_total+1)
        return loss


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distance_func = nn.MSELoss(reduction = 'sum') # mean squared error for comparing two PSDs

    def forward(self, postive1, postive2):
        pos_loss = self.distance_func(postive1, postive2) / len(postive1)
        if len(postive1) == 1:
            return pos_loss
        else:
            neg_loss = 0.
            M = 0
            for i in range(len(postive1)):
                for j in range(len(postive2)):
                    if i != j:
                        neg_loss += self.distance_func(postive1[i], postive2[j])
                        M += 1
            neg_loss = -1 * neg_loss / M

            return pos_loss + neg_loss