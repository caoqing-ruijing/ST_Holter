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

class Noise_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, leads=None, n_max_cls=3, date_len=5000,random_crop=False,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            InputSameLabel: chose noise both the input and label, specious training
            random_sampling: random choise input
        """
        self.dataframe = self.__select_lead_df(csv_file,leads)
        self.transform = transform
        self.date_len = date_len
        self.random_crop = random_crop
        self.n_max_cls = n_max_cls

    def __len__(self):
        return len(self.dataframe)


    def __select_lead_df(self,csv_file,leads):
        '''
        选择指定lead
        '''
        df = pd.read_csv(csv_file)
        df_s = []
        for lead in leads:
            df_s.append(df[df['lead']==lead])
        dataframe = pd.concat(df_s)
        return dataframe

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_file = os.path.join(self.dataframe.iloc[idx]['paths'])
        lead = os.path.join(self.dataframe.iloc[idx]['lead'])
        
        if '.feather' in data_file:
            data_feather = pd.read_feather(data_file)
        elif '.csv' in data_file:
            data_feather = pd.read_csv(data_file)

        input_i = data_feather['{}_value'.format(lead)]
        label_i = data_feather['{}_label'.format(lead)]

        if self.date_len < len(input_i):
            if self.random_crop is True:
                start_pt = random.randint(10,len(input_i)-self.date_len-10)
                end_pt = start_pt+self.date_len
            else:
                gap = int((len(input_i)-self.date_len)/2)
                start_pt = gap
                end_pt = gap+self.date_len

            input_i = input_i[start_pt:end_pt]
            label_i = label_i[start_pt:end_pt]

        elif self.date_len > len(input_i):
            assert 1>2,'date_len {} > len(input_i) {}'.format(self.date_len, len(input_i))

        input_i = np.array(input_i).astype('float32')
        label_i = np.array(label_i).astype('int64').clip(0,self.n_max_cls)
        label_i_onehot = np.eye(self.n_max_cls)[label_i].astype('int64')
        label_i_onehot = np.transpose(label_i_onehot,(1,0))  # 5000,3 -> 3,5000 to match the output B,C,L

        # shape (length) -> (channel, length)
        input_i = input_i.reshape(1, input_i.shape[0]) 
        # label_i = label_i.reshape(1, label_i.shape[0]) 

        sample = {'input': input_i, 'label': label_i, 'label_onehot':label_i_onehot}

        if self.transform:
            sample = self.transform(sample)

        return sample

        # return input_i, label_i

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, sample):
        input_i, label_i,label_onehot = sample['input'], sample['label'],sample['label_onehot']
        # input_i = input_i.transpose((2, 0, 1)) h w c -> c h w
        return {'input': torch.from_numpy(input_i),
                'label': torch.from_numpy(label_i),
                'label_onehot': torch.from_numpy(label_onehot)}


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

