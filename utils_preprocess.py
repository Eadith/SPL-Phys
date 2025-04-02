from mtcnn.mtcnn import MTCNN
import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
import shutil
import h5py
import pandas as pd
from scipy.interpolate import CubicSpline

detector = MTCNN()


def read_avi(video_path):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_frames = np.zeros((num_frames, frame_height, frame_width, 3),dtype=np.uint8)
    
    image_id = 0
    while cap.isOpened():
        # curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imwrite(os.path.join(new_path,image_name), frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames[image_id] = frame
        image_id += 1

    cap.release()

    return video_frames, num_frames, frame_height, frame_width, fps

def read_imgs(imgs_path):
    imgs_files = os.listdir(imgs_path)
    frames = []
    for i in range(1, len(imgs_files) + 1):
        img_path = imgs_path + os.sep + '%04d.jpg'%(i)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    frames = np.array(frames)
    return frames, len(frames)




def v1_process(dataset_path='path to dataset',save_root='path for save croped imgs', img_size=128, use_larger_box=True, larger_box_coef=1.5):
    videos_path = glob.glob(os.path.join(dataset_path,'*', '*', '*'))

    if not os.path.exists(save_root):
            os.makedirs(save_root)

    for v_path in videos_path:
        v_path_list = v_path.split('\\')
        # filter out NIP (source4)
        
        save_path = save_root + '/' + '_'.join(v_path.split('\\')[-3:]) + '.h5'

        # 读取ppg和fps信息
        fps = 30
        ppg = pd.read_csv(os.path.join(r'E:\vipl-hr\unzip', v_path_list[-3], v_path_list[-2], v_path_list[-1], 'wave.csv' )) #TODO
        ppg = np.array(ppg)[:,0]

        # read video to numpy
        # video_frames, num_frames, frame_height, frame_width, _ = read_avi(video_path=v_path)

        video_frames, num_frames = read_imgs(v_path)

        # croped imgs
        video_h5 = np.zeros((num_frames, img_size, img_size, 3), dtype=np.uint8)
        # crop face each clip
        # coords
        frame = video_frames[0]
        detections = detector.detect_faces(frame)
        
        if len(detections) > 0:
            # Pick the highest score
            for det in detections:
                if det['confidence'] >= 0.9:
                    face_box_coor = det['box']
        else:
            print("ERROR: No Face Detected {}".format(v_path))
            face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        
        face_region = np.asarray(face_box_coor, dtype='int')

        for i in range(num_frames):
            frame = video_frames[i]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            tmp = cv2.resize(frame, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
            video_h5[i] = tmp


        with h5py.File(save_path, 'w') as f:
            data = f.create_dataset('video', shape=(num_frames, img_size, img_size, 3), 
                                        dtype='uint8', chunks=(1,img_size, img_size,3),
                                        compression="gzip", compression_opts=4, data=video_h5)
            sigs = f.create_dataset("bvp", data=np.array(ppg), dtype='f')
            freqs = f.create_dataset("fps", data=[fps])
            
        print('-----------------------------> done!')