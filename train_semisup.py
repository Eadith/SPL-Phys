import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import json
import torch
from PhysNetModel import PhysNet

from utils_negpearson import Neg_Pearson
from utils_MCCs import NegativeMaxCrossCorr
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_datasets import *
from utils_data import VIPL_split_percentage, UBFC_LU_split, PURE_split, MMSE_split_percentage
from utils_sig import *

from utils_pseudo import *
from utils_corssentropy import TorchLossComputer
from utils_frequencyloss import ContrastiveLoss

from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    
# torch.autograd.set_detect_anomaly(True)

@ex.config
def my_config():
    # here are some hyperparameters in our method
    seed = 101
    # hyperparams for model training
    total_epoch = 30 # total number of epochs for training the model
    lr = 1e-4 # learning rate
    bs = 4 # default = 2 for contrast loss
    mu = 1 # batch size multiple of unlabel data.
    num_workers = 4

    # hyperparams for ST-rPPG block
    fs = 30 # video frame rate, TODO: modify it if your video frame rate is not 30 fps.
    T = 150 # temporal dimension of ST-rPPG block, default is 10 seconds.
    
    reuse = False # if use the pretrained weight released from contrast-phys+
    label_ratio = 0.5 # TODO: if you dataset is fully labeled, you can set how many labels are used for training.
    threshold = 0.6 # snr的值，超过则表示伪标签可以用于监督其增强样本
    temperature = 0.1 # SupConLoss 损失的温度超参数

    fold = 5
    lambda1 = 10
    lambda2 = 1
    interval = 10 # # evaluate 10s clips 

    echo = 10
    train_exp_name = 'default'

    print(train_exp_name)
    result_dir = '/root/%s'%(train_exp_name) # store checkpoints and training recording
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, seed, total_epoch, T, lr, bs, mu, result_dir, fs, label_ratio, reuse, num_workers, threshold, temperature, dropout, interval, fold, lambda1, lambda2, echo):

    # 设置Numpy和PyTorch的随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 如果使用CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # 5折划分
    # train_list, test_list = PURE_split()
    # train_list, test_list = UBFC_LU_split()
    train_list, test_list = VIPL_split_percentage(k=5, idx=fold)
    train_list_label = train_list[0:int(len(train_list) * label_ratio)]
    train_list_unlabel = train_list[int(len(train_list) * label_ratio):]
    # 保存训练与验证数据路径
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/test_list.npy', test_list)

    # define the dataloader
    labeled_dataset = H5Dataset(train_list_label, T) # please read the code about H5Dataset when preparing your dataset
    unlabeled_dataset = H5Dataset_u(train_list_unlabel, T)

    labeled_dataloader = DataLoader(labeled_dataset, batch_size=bs, # two videos for contrastive learning
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True) # TODO: If you run the code on Windows, please remove num_workers=4.
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=bs*mu, # two videos for contrastive learning
                            shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    # validation dataloader
    val_dataset = H5Dataset_val(test_list)
    val_dataloader = DataLoader(val_dataset, batch_size=1, # 1 video for validation or testing
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # define the model and loss
    model = PhysNet()
    
    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model).to(device)  # 将模型对象转变为多GPU并行运算的模型
    else:
        model = model.to(device)
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
    
    # 监督损失使用Pearson、mccs、···
    sup_loss_func_pear = Neg_Pearson(device)
    # 监督MCCs损失
    sup_loss_func_mcc = NegativeMaxCrossCorr(fs, high_pass=40, low_pass=250)

    # 无监督损失函数使用contrastloss、稀疏性损失、····
    unsup_loss_func_con = ContrastiveLoss()

    
    # 信号质量评价函数，ipr、snr、···
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.Adam(model.parameters(), lr=lr)
    
    # 损失的超参数，pseudo损失线性增大 0 --> 1
    init_alpha = 0.6
    max_alpha = 0.5
    alpha_step = (max_alpha - init_alpha) / total_epoch
    count = 0
    for e in range(total_epoch):
        current_alpha = init_alpha + alpha_step*e
        total_L = []
        sup_L = []
        unsup_L1 = []
        unsup_L2 = []
        PNR = []
        PNR_std = []

        labeled_train_iter = iter(labeled_dataloader)
        unlabeled_train_iter = iter(unlabeled_dataloader)
        train_iteration = len(unlabeled_train_iter) + len(labeled_train_iter)
        model.train()
        for it in range(1): # TODO: 每轮中每条视频的循环次数

            for batch_idx in range(train_iteration):
                try:
                    imgs, GT_sig, fps = labeled_train_iter.next()
                except: # 当循环结束时，重新开始循环
                    labeled_iter = iter(labeled_dataloader)
                    imgs, GT_sig, fps = labeled_iter.next()

                try:        
                    imgs_unlabel, imgs_aug_unlabel, imgs_hard_aug_unlabel, fps_unlabel = unlabeled_train_iter.next()
                except: # 当循环结束时，重新开始循环
                    unlabeled_iter = iter(unlabeled_dataloader)
                    imgs_unlabel, imgs_aug_unlabel, imgs_hard_aug_unlabel, fps_unlabel = unlabeled_iter.next()
                # 加载到cuda
                imgs, GT_sig, fps = imgs.to(device), GT_sig.to(device), fps.to(device)
                imgs_unlabel, imgs_aug_unlabel, imgs_hard_aug_unlabel, fps_unlabel = imgs_unlabel.to(device), imgs_aug_unlabel.to(device), imgs_hard_aug_unlabel.to(device), fps_unlabel.to(device)

                rppg = model(imgs)
                rppg_u = model(imgs_unlabel)
                rppg_a_u = model(imgs_aug_unlabel)
                rppg_s_u = model(imgs_hard_aug_unlabel)


                # 根据label_flag区分监督损失还是伪标签学习

                # 监督pearson/MCCs损失
                sup_loss = sup_loss_func_mcc(rppg, GT_sig).mean() #
                # sup_loss = sup_loss_func_pear(rppg, GT_sig).mean()


                # 划分无标签样本
                mask = torch.zeros(bs*mu).to(device)
                pseudo_hrs = []
                pnr_batch = []
                for idx in range(bs*mu):
                    freqs, psd = torch_power_spectral_density(rppg_u[idx].view(1,-1).clone().detach(), fps=fps_unlabel[idx], normalize=True, bandpass=True)
                    freqs_aug, psd_aug = torch_power_spectral_density(rppg_a_u[idx].view(1,-1).clone().detach(), fps=fps_unlabel[idx], normalize=True, bandpass=True)
                    max_probs, max_idx = torch.max(psd, dim=1)
                    max_idx = max_idx.detach()
                    pseudo_hrs.append(freqs[max_idx] * 60)

                    pnr1 = PNR(freqs, psd, device=device, bandpassed=False)
                    pnr = pnr1.item()
                    pnr_batch.append(pnr)
                    if pnr >= threshold:
                        mask[idx] = 1
                PNR.append(sum(pnr_batch) / len(pnr_batch))
                PNR_std.append(np.std(pnr_batch))
                mask_idx = np.where(mask.cpu() == 0)[0]
                if len(mask_idx) == 0:
                    unsup_loss1 = torch.from_numpy(np.array(0))
                else:
                    # 使用对比损失约束
                    _, psd1 = torch_power_spectral_density(rppg_u[mask_idx], fps=fs, bandpass=True, normalize=True)
                    _, psd2 = torch_power_spectral_density(rppg_a_u[mask_idx], fps=fs, bandpass=True, normalize=True)
                    unsup_loss1 = unsup_loss_func_con(psd1, psd2)

                # 进行伪标签学习 1,mcc 2,交叉熵
                pseudo_rppg = rppg_u.clone().detach() + rppg_a_u.clone().detach()
                for i in range(len(pseudo_rppg)):
                    filted_pseudo_rppg = butter_bandpass(pseudo_rppg[i].cpu().numpy(), 0.66, 4.16, fps_unlabel[i].cpu().numpy())
                    pseudo_rppg[i] = torch.from_numpy(filted_pseudo_rppg.copy()).to(device)
                unsup_loss_mcc = sup_loss_func_mcc(rppg_s_u, pseudo_rppg) * mask
                unsup_loss2 = unsup_loss_mcc.mean()

                # 交叉熵损失
                # unsup_loss2 = torch.tensor(0.0, device=device)
                # unmask_idx = np.where(mask.cpu() == 1)[0]
                # if len(unmask_idx) > 0:
                #     for idx in unmask_idx:
                #         _,pseudo_loss,_ = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(rppg_s_u[idx].view(1,-1), pseudo_hrs[idx].to(device), fps_unlabel[idx])
                #         unsup_loss2 += pseudo_loss
                #     unsup_loss2 = unsup_loss2 / len(unmask_idx)

                # total loss
                loss = sup_loss + lambda1 * unsup_loss1 + lambda2 * unsup_loss2

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()


                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("sup_loss", sup_loss.item())
                ex.log_scalar("unsup_loss1", unsup_loss1.item())
                ex.log_scalar("unsup_loss2", unsup_loss2.item())

                
        # save model checkpoints
        torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)