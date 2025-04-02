import numpy as np
import os
import cv2
import h5py
import random
import torch
from torch.utils.data import Dataset
from scipy import signal

from augmentation import *


b, a = signal.butter(4, [2 * 0.7 / 30, 2 * 4.0 / 30], 'bandpass')

def resample_ppg(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(
        np.linspace(
            1, input_signal.shape[0], target_length), np.linspace(
            1, input_signal.shape[0], input_signal.shape[0]), input_signal)

def augment(img_in):
    info_aug = {'flip_h': False, 'flip_v': False, 'rot0': False, 'rot90': False,'rot180': False,'rot270': False}
    randomaug=random.sample(['flip_h', 'flip_v','rot90','rot180','rot270'],1)
    info_aug[randomaug[0]]=True

    if info_aug['flip_v']:
        img_in = [cv2.flip(j,0) for j in img_in]

    if info_aug['flip_h']:
        img_in = [cv2.flip(j,1) for j in img_in]

    if info_aug['rot90']:
        img_in = [cv2.rotate(j,cv2.ROTATE_90_CLOCKWISE) for j in img_in]
    if info_aug['rot270']:
        img_in = [cv2.rotate(j,cv2.ROTATE_90_COUNTERCLOCKWISE) for j in img_in]
    if info_aug['rot180']:
        img_in = [cv2.rotate(j,cv2.ROTATE_180) for j in img_in]

    return img_in

def augment_hard(img_in):
    """
        降低视频分辨率，同时随机裁剪；
    """
    h, w = img_in.shape[1], img_in.shape[2]
    img_out = []
    for i in img_in:
        img = cv2.GaussianBlur(i, (0,0), 1, 1)
        image_2x = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) 
        img = cv2.resize(image_2x, (w,h), interpolation=cv2.INTER_LINEAR) 
        img_out.append(img)

    info_aug = {'LU': False, 'LD': False, 'RU': False, 'RD': False,'CC': False}
    randomaug=random.sample(['LU', 'LD','RU','RD'],1)
    info_aug[randomaug[0]]=True

    if info_aug['LU']:
        img_out = [cv2.resize(j[:96,:96,:], (w,h), interpolation=cv2.INTER_LINEAR) for j in img_out]

    if info_aug['LD']:
        img_out = [cv2.resize(j[32:,:96,:], (w,h), interpolation=cv2.INTER_LINEAR) for j in img_out]

    if info_aug['RD']:
        img_out = [cv2.resize(j[32:,32:,:], (w,h), interpolation=cv2.INTER_LINEAR) for j in img_out]

    if info_aug['RU']:
        img_out = [cv2.resize(j[:96,32:,:], (w,h), interpolation=cv2.INTER_LINEAR) for j in img_out]

    return img_out

# def built_sets(sets_list, label_ratio, dataset_root):

#     video_list = []
#     for dataset in sets_list:
#         video_paths = os.listdir(dataset_root + os.sep + dataset)

#         if dataset == "UBFC":
#             video_paths = [x for x in video_paths if x[7:-3] not in ubfc_val_paths]
#         elif dataset == "PURE":
#             video_paths = [x for x in video_paths if x[:3] not in pure_val_paths]
#         elif dataset == 'VIPL-V1':
#             video_paths = [x for x in video_paths if x.split('_')[0] not in vipl_val_paths]
#         else:
#             print('not support for training!')

#         for path in video_paths:
#             video_list.append(dataset_root + os.sep + dataset + os.sep + path)

#     video_list = np.random.permutation(video_list)

#     label_video_num = int(len(video_list) * label_ratio)

#     label_video_list = video_list[:label_video_num]
#     try:
#         unlabel_video_list = video_list[label_video_num:]
#     except:
#         unlabel_video_list = []

#     return label_video_list, unlabel_video_list

class H5Dataset_val(Dataset):

    def __init__(self, train_list):
        # TODO: Please note that the following code in __init__ is for fully labeled dataset. We manually set label_ratio to control how many labels are used for training.
        # TODO: if some videos have labels while others not in your dataset, please make sure the the labeled videos are all at the front of the self.train_list, and self.label_sample_number is the number of labeled videos.
        self.train_list = train_list # list of .h5 file paths for training
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        with h5py.File(self.train_list[idx], 'r') as f:
            name = self.train_list[idx].split('/')[-1]
            img_length = f['video'].shape[0]
            try:
                bvp = resample_ppg(f['bvp'][:], img_length)
            except:
                bvp = resample_ppg(f['ppg'][:], img_length)
            
            try:
                fps = f['fs'][0]
            except:
                fps = f['fps'][0]
                
            # using full length for validation
            img_seq = f['video']
            label = bvp.astype('float32')

            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32') / 255.0

        return img_seq, label, fps, name
    

class H5Dataset(Dataset):

    def __init__(self, train_list, T=None, transform=None):
        # TODO: Please note that the following code in __init__ is for fully labeled dataset. We manually set label_ratio to control how many labels are used for training.
        # TODO: if some videos have labels while others not in your dataset, please make sure the the labeled videos are all at the front of the self.train_list, and self.label_sample_number is the number of labeled videos.
        self.train_list = train_list # list of .h5 file paths for training

        self.T = T # video clip length
        self.transform = transform
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['video'].shape[0]
            try:
                bvp = resample_ppg(f['bvp'][:], img_length)
            except:
                bvp = resample_ppg(f['ppg'][:], img_length)
                
            try:
                fps = f['fs'][0]
            except:
                fps = f['fps'][0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            label = bvp[idx_start:idx_end].astype('float32')

            img_seq = f['video'][idx_start:idx_end]
            
            if random.random() > 0.7:
                img_seq = np.array(augment(img_seq))


            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32') / 255.0
            # data_aug = np.transpose(data_aug, (3, 0, 1, 2)).astype('float32') / 255.0
            # if self.transform:
            #     input = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data]))
            #     input_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_aug.copy()]))
            #     input_hard_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_hard_aug.copy()]))

        return img_seq, label, fps

class H5Dataset_u(Dataset):

    def __init__(self, train_list, T, wp=0.7, sp=0.5):
        # TODO: Please note that the following code in __init__ is for fully labeled dataset. We manually set label_ratio to control how many labels are used for training.
        # TODO: if some videos have labels while others not in your dataset, please make sure the the labeled videos are all at the front of the self.train_list, and self.label_sample_number is the number of labeled videos.
        self.train_list = train_list # list of .h5 file paths for training

        self.T = T # video clip length
        self.transform = np.random.binomial(1, wp, size=(T, 128, 128, 3))
        self.transform_hard = np.random.binomial(1, sp, size=(T, 128, 128, 3))
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['video'].shape[0]
            # try:
            #     bvp = resample_ppg(f['bvp'][:], img_length)
            # except:
            #     bvp = resample_ppg(f['ppg'][:], img_length)
            try:
                fps = f['fs'][0]
            except:
                fps = f['fps'][0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            # label = bvp[idx_start:idx_end].astype('float32')

            img_seq = f['video'][idx_start:idx_end]

            data_aug = np.array(augment(img_seq))
            data_hard_aug = np.array(augment_hard(img_seq))
            # data_aug = img_seq * self.transform
            # data_hard_aug = img_seq * self.transform_hard

            img_seq = img_seq.astype('float32') / 255.0
            data_aug = data_aug.astype('float32') / 255.0
            data_hard_aug = data_hard_aug.astype('float32') / 255.0

            # data_hard_aug = video_color_jitter(img_seq)



            img_seq = np.transpose(img_seq, (3, 0, 1, 2))
            data_aug = np.transpose(data_aug, (3, 0, 1, 2))
            data_hard_aug = np.transpose(data_hard_aug, (3, 0, 1, 2))
            
            # if self.transform:
            #     input = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data]))
            #     input_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_aug.copy()]))
            #     input_hard_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_hard_aug.copy()]))

        return img_seq, data_aug, data_hard_aug, fps
    

class H5Dataset_(Dataset):

    def __init__(self, train_list, T, label_ratio, transform=None, transform_hard=None):
        # TODO: Please note that the following code in __init__ is for fully labeled dataset. We manually set label_ratio to control how many labels are used for training.
        # TODO: if some videos have labels while others not in your dataset, please make sure the the labeled videos are all at the front of the self.train_list, and self.label_sample_number is the number of labeled videos.
        self.train_list = train_list # list of .h5 file paths for training

        self.T = T # video clip length
        self.label_sample_number = int(len(self.train_list) * label_ratio)
        self.transform = transform
        self.transform_hard = transform_hard
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if idx < self.label_sample_number:
            label_flag = np.float32(1)
        else:
            label_flag = np.float32(0)

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['video'].shape[0]
            try:
                bvp = resample_ppg(f['bvp'][:], img_length)
            except:
                bvp = resample_ppg(f['ppg'][:], img_length)
            try:
                fps = f['fs'][0]
            except:
                fps = f['fps'][0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            label = bvp[idx_start:idx_end].astype('float32')

            img_seq = f['video'][idx_start:idx_end]
            
            if label_flag:
                data_aug = augment(img_seq)
            else:
                data_aug = augment_hard(img_seq)

            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            data_aug = np.transpose(data_aug, (3, 0, 1, 2)).astype('float32')
            # if self.transform:
            #     input = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data]))
            #     input_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_aug.copy()]))
            #     input_hard_aug = torch.tensor(np.array([self.transform(j).detach().numpy() for j in data_hard_aug.copy()]))

        return img_seq, data_aug, label, label_flag, fps
    


if __name__ == '__main__':
    from torchvision.transforms import Compose, ToTensor, Normalize
    from utils_datasets import *
    from torch.utils.data import DataLoader

    def transform():
        return Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
    paths = [r'G:\3rd-ReSPP-Track_1\ubfc\subject1.h5', r'G:\3rd-ReSPP-Track_1\ubfc\subject4.h5', r'G:\3rd-ReSPP-Track_1\ubfc\subject3.h5']
    datasets = H5Dataset(paths, 150, transform=transform())
    a, b, c = datasets[0]
    print(a.shape)

    labeled_dataloader = DataLoader(datasets, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=0, pin_memory=True, drop_last=True) 
    for x, y, z in labeled_dataloader:
        print(z.shape)
        break