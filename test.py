import numpy as np
import h5py
import torch
import os
import pandas as pd
from PhysNetModel import PhysNet_contrast
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json
from utils_post_process import *
from utils_pseudo import *

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 26 # the model checkpoint at epoch 
    train_exp_name = 'default'
    train_exp_num = 3 # the training experiment number
    train_exp_dir = './%s/%d'%(train_exp_name, train_exp_num) # training experiment directory
    time_interval = 10 # evaluate 30s clips

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval):

    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    
    for e in range(30):
        model = PhysNet_contrast(config_train['S'], config_train['in_ch']).to(device).eval()
        from collections import OrderedDict
        state_dict = torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)  # 当前路径 model 文件下
        new_state_dict = OrderedDict()   # create new OrderedDict that does not contain `module.`
        for k, v in state_dict.items():  # remove `module.`
            name = k[7:]                 # 或 name = k.replace('module.', '')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)

        # model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)) # load weights to the model

        @torch.no_grad()
        def dl_model(imgs_clip):
            # model inference
            img_batch = imgs_clip
            img_batch = img_batch.transpose((3,0,1,2))
            img_batch = img_batch[np.newaxis].astype('float32') / 255.0
            img_batch = torch.tensor(img_batch).to(device)

            rppg = model(img_batch)[:,-1, :]
            rppg = rppg[0].detach().cpu().numpy()
            return rppg

        gt_hr = []
        pd_hr = []

        for h5_path in test_list:
            h5_path = str(h5_path)

            with h5py.File(h5_path, 'r') as f:
                imgs = f['video']
                fs = int(f['fs'][0])

                duration = imgs.shape[0] // fs
                num_blocks = int(duration // time_interval)

                rppg_list = []

                for b in range(num_blocks):
                    rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
                    rppg_list.append(rppg_clip)

                rppg_list = np.array(rppg_list) #TODO pearson loss

                rppg = np.concatenate(rppg_list, axis=0)

                # normalized
                pred = calculate_metric_per_video(rppg, fs, diff_flag=False,use_bandpass=True,hr_method='FFT')
                gt = calculate_metric_per_video(f['ppg'][:], 60, diff_flag=False,use_bandpass=True,hr_method='FFT')
                # gt = hr
                pd_hr.append(pred)
                gt_hr.append(gt)

        # calculate metrics
        predict_hr_fft_all = np.array(pd_hr)
        gt_hr_fft_all = np.array(gt_hr)

        # MAE metric
        num_test_samples = len(predict_hr_fft_all)
        MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
        mae_standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)

        RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
        rmse_standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)

        MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
        mape_standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100

        Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
        correlation_coefficient = Pearson_FFT[0][1]
        p_standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
        print("Epoch: %d ==> MAE: %.4f +/- %.4f  RMSE: %.4f +/- %.4f  MAPE: %.4f +/- %.4f  Pearson: %.4f +/- %.4f \n"%(e,MAE_FFT, mae_standard_error, RMSE_FFT, rmse_standard_error, MAPE_FFT, mape_standard_error, correlation_coefficient, p_standard_error))
