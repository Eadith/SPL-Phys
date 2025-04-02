# PulseMatch



This is the main code repository of the paper "PulseMatch: Towards Semi-Supervised Deep Remote Photoplethysmography Measurement from Facial Video" for semi-supervised rPPG training.


## Dataset Preprocessing
We use MTCNN to detect and crop video clip. And the processed data is stored by h5 file.
Since semi-supervised may use limited labels for training, you should add bvp for your labeled video files in the training set.



## Training and testing

### Training
The main training file is `train_semisup.py`. The data split function in `utils_data.py` and  dataloader files in `utils_datasets.py`. You can modify the data path and directly run the file `train_semisup.py`.
 More detailed code will be updated after the paper is published.

```
python train_semisup.py
```
When you first run `train_semisup.py`, the training recording including model weights and some metrics will be saved at `./default/1`

### Testing

After training, you can test the model on the test set. Please make sure .h5 files in test set have `bvp`. You can directly run
```
python test.py with train_exp_num=1
```
The predicted rPPG signals and ground truth PPG signals are saved in `./results/1/1`. You can filter the rPPG and bvp signals by `butter_bandpass` function with lowcut=0.75 and highcut=3 and get heart rates by `calculate_metric_per_video` function in `utils_post_process.py`. 