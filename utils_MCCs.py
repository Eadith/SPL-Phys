import torch
import torch.nn as nn
tr = torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft


class NegativeMaxCrossCov(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels):
        # Normalize
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)
        
        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:,use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:,zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        # Zero out irrelevant freqs
        X[:,zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio
        
        return -max_cc


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = tr.std(preds, dim=-1) * tr.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        return output 


if __name__ == '__main__':
    y = torch.randn(4,160).cuda()
    y1 = torch.rand(4,160).cuda()
    macc = NegativeMaxCrossCorr(30,30,180).cuda()
    loss = macc(y,y)
    print(loss)