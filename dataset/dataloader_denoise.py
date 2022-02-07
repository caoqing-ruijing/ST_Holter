import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
# from torch._C import device
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import warnings
warnings.filterwarnings("ignore")

from scipy.signal import find_peaks

def random_crop_noise(bw_noise_full,input_len=5000,flat_top_n=10):
    start_pt = random.randint(10,len(bw_noise_full)-input_len-10)
    end_pt = start_pt+input_len

    bw_noise = bw_noise_full[start_pt:end_pt]
    assert bw_noise.shape[0] == input_len

    # flat_top_n = 0
    r_peaks= []
    if flat_top_n >0:
        r_peaks, properties = find_peaks(bw_noise,
                                         distance = 100, #间隔0.5 150
                                        )
        qrs_length = 0.05*500
        for r_peak in r_peaks:
            before = r_peak-int(qrs_length/2)
            after = r_peak+int(qrs_length/2)+1
            bw_noise[before:after]=0
    
    if random.uniform(0,1) > 0.6:
        n_noise_chunk = random.randint(1,6)
        bw_noise_raw = np.array([0]*len(bw_noise))
        for i in range(n_noise_chunk):
            chunk_size = random.randint(300,600) 
            start_pt = random.randint(10,len(bw_noise_raw)-chunk_size-10)
            end_pt = start_pt+chunk_size
            bw_noise_raw[start_pt:end_pt] = bw_noise[start_pt:end_pt]
        
        return bw_noise_raw
    else:
        return bw_noise


class Noise_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, normal_csv, noise_csv, n_sample=10000, 
                leads=None, input_len=5000,max_syn_noise=3,
                add_noise_ratio=0.5,random_ranges=[0.2,0.5],
                transform=None,train_mode=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            InputSameLabel: chose noise both the input and label, specious training
            random_sampling: random choise input
        """
        self.transform = transform
        self.leads = leads
        normal_df = pd.read_csv(normal_csv)
        self.dataframe = pd.concat([normal_df,normal_df,normal_df,normal_df,normal_df,normal_df]).reset_index()

        self.noise_paths = pd.read_csv(noise_csv[0])['paths'].tolist()
        self.mit_paths_csvs = self._load_MIT_csv(noise_csv)
        
        self.max_syn_noise = max_syn_noise
        self.n_sample = n_sample
        self.input_len = input_len
        self.add_noise_ratio = add_noise_ratio
        self.random_ranges = random_ranges
        self.train_mode = train_mode
        # if train_mode is False:
            # random.seed(10)

    def __len__(self):
        return len(self.dataframe)
        # return self.n_sample

    def _load_MIT_csv(self,noise_csv):
        bw_path,em_path,ma_path = noise_csv[1],noise_csv[2],noise_csv[3]
        bw_paths = pd.read_csv(bw_path)['paths'].tolist()
        em_paths = pd.read_csv(em_path)['paths'].tolist()
        ma_paths = pd.read_csv(ma_path)['paths'].tolist()
        return [bw_paths,em_paths,ma_paths]


    def _read_split_feather(self,noise_csv):
        noise_df = pd.read_feather(noise_csv)
        split_n = int(len(noise_df)/10)

        # for i in range(0,len(noise_df),split_n):
        split_idx = []
        for i in range(0,len(noise_df),split_n):
            split_idx.append(i)

        split_batch = []
        for j in range(len(split_idx)):
            if j == len(split_idx)-1:
                split_batch.append([split_idx[j],-1])
            else:
                split_batch.append([split_idx[j],split_idx[j+1]])

        noise_df_splits = []
        for split_batch_i in split_batch:
            noise_df_i = noise_df[split_batch_i[0]:split_batch_i[1]]
            noise_df_splits.append(noise_df_i)
        return noise_df_splits

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file = os.path.join(self.dataframe.iloc[idx]['paths'])
        lead = random.choice(self.leads)
        data_feather = pd.read_feather(data_file)
        data_feather_i = data_feather[lead]
        # print(data_feather_i)

        if self.train_mode is True:
            start_pt = random.randint(1, len(data_feather_i)-self.input_len-1)
            end_pt = start_pt+self.input_len
        else:
            start_pt = 0
            end_pt = start_pt+self.input_len
    
        input_i = data_feather_i[start_pt:end_pt]
        label_i = data_feather_i[start_pt:end_pt]
        default_input_i = data_feather[lead+'_input'][start_pt:end_pt]

        input_i = np.array(input_i).astype('float32')
        label_i = np.array(label_i).astype('float32')
        default_input_i = np.array(default_input_i).astype('float32')
        # print('self.train_mode',self.train_mode,'default_input_i',default_input_i)

        assert len(input_i) == self.input_len
        assert len(label_i) == self.input_len

        if self.train_mode is True:

            if random.uniform(0,1)>self.add_noise_ratio:
                if random.uniform(0,1)>0.7:
                    assert len(input_i) == self.input_len,'input_i {} VS input_len {}'.format(len(input_i), self.input_len)
                    # for noise_paths in ([MIT_noise_paths_bw,MIT_noise_paths_em,MIT_noise_paths_ma]):
                    for noise_paths in self.mit_paths_csvs:
                        noise_path_i = random.choice(noise_paths)
                        noise_feather_i = pd.read_feather(noise_path_i)[lead]
                        if len(noise_feather_i)>len(input_i):
                            noise_crop = random_crop_noise(noise_feather_i,input_len=len(input_i),flat_top_n=30)
                        elif len(noise_feather_i)==len(input_i):
                            noise_crop = noise_feather_i
                        else:
                            noise_feather_i = pd.concat([noise_feather_i,noise_feather_i,noise_feather_i]).reset_index(drop=True)
                            noise_crop = random_crop_noise(noise_feather_i,input_len=len(input_i),flat_top_n=30)
                        # noise_crop = random_crop_noise(noise_feather_i,input_len=len(input_i),flat_top_n=0)
                        try:
                            noise_crop = noise_crop.reset_index(drop=True)
                        except:
                            pass

                        ratio1 = random.uniform(0.2,0.5)
                        input_i = input_i+ratio1*noise_crop
                        assert len(input_i) == self.input_len,'input_i {} VS input_len {}'.format(len(input_i), self.input_len)
                else:
                    max_n = random.randint(1, self.max_syn_noise)
                    for n_chose in range(max_n):
                        noise_path = random.choice(self.noise_paths)
                        noise_feather_i = pd.read_feather(noise_path)[lead]
                        # noise_feather_i = noise_df[lead]

                        # flat_top_n = random.randint(0, 9)
                        # noise_crop = random_crop_noise(noise_feather_i,input_len=self.input_len,flat_top_n=20)
                        if len(noise_feather_i)>len(input_i):
                            noise_crop = random_crop_noise(noise_feather_i,input_len=len(input_i),flat_top_n=30)
                        elif len(noise_feather_i)==len(input_i):
                            noise_crop = noise_feather_i
                        else:
                            noise_feather_i = pd.concat([noise_feather_i,noise_feather_i,noise_feather_i]).reset_index(drop=True)
                            noise_crop = random_crop_noise(noise_feather_i,input_len=len(input_i),flat_top_n=30)

                        noise_crop = np.array(noise_crop).astype('float32')
                        # print('noise_crop',noise_crop.shape)
                        # print('input_i',input_i.shape)
                        # ratio1 = random.uniform(0.2,0.5)
                        ratio1 = random.uniform(self.random_ranges[0],self.random_ranges[1])
                        input_i = input_i+ratio1*noise_crop
                    assert len(input_i) == self.input_len,'input_i {} VS input_len {}'.format(len(input_i), self.input_len)

            # else:
            #     input_i = default_input_i
        else:
            assert len(input_i) == self.input_len,'input_i {} VS input_len {}'.format(len(input_i), self.input_len)
            # print('val mode')
            input_i = default_input_i

        assert len(input_i) == self.input_len,'input_i {} VS input_len {}'.format(len(input_i), self.input_len)
        assert len(label_i) == self.input_len,'label_i {} VS input_len {}'.format(len(label_i), self.input_len)
        
        input_i = np.array(input_i).astype('float32')
        label_i = np.array(label_i).astype('float32')

        # shape (length) -> (channel, length)
        input_i = input_i.reshape(1, input_i.shape[0]) 
        label_i = label_i.reshape(1, label_i.shape[0])

        sample = {'input': input_i, 'label': label_i}
        if self.transform:
            sample = self.transform(sample)
        return sample
        # return input_i, label_i

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, sample):
        input_i, label_i = sample['input'], sample['label']
        # input_i = input_i.transpose((2, 0, 1)) h w c -> c h w
        return {
                'input': torch.from_numpy(input_i),
                'label': torch.from_numpy(label_i),
        }


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

