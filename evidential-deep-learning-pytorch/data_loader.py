import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class MNIST():
    def __init__(self, input_path:str):
        self.df = pd.read_csv(input_path)
        print(f'Loaded data, {self.df.shape[0]} samples')
        self.labels = self.get_label()
        self.inputs = self.get_input()
        self.normalize_inputs()
        print('Loaded input data')
        print(f'Input data shape = {self.inputs.shape}')
        print(f'Output data shape = {self.labels.shape}')
        self.df = None

    def get_label(self):
        return self.df['label'].values
    
    def get_input(self):
        columns = self.df.columns.tolist()
        columns.remove('label')
        return self.df[columns].values

    def normalize_inputs(self):
        self.inputs = self.inputs/255


class MNISTLoader(Dataset):
    def __init__(self, data: MNIST):
        super().__init__()
        self.data = data
        self.labels = self.data.labels
        self.inputs = self.data.inputs
        self.n_samples = self.inputs.shape[0]
        self.input_dim = self.inputs.shape[1]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):

        x = self.inputs[index]
        y = self.labels[index]
        return torch.FloatTensor(x), y
    
    @staticmethod
    def collate():
        pass


class EDL_MNISTLoader(Dataset):
    def __init__(self, data: MNIST):
        super().__init__()
        self.data = data
        self.labels = self.data.labels
        self.inputs = self.data.inputs
        self.n_samples = self.inputs.shape[0]
        self.input_dim = self.inputs.shape[1]
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):

        x = self.inputs[index]
        y = self.labels[index]
        y = self.to_categorical(y)
        return torch.FloatTensor(x), y
    
    def to_categorical(self, y, num_class = 10):
        to_cat = np.eye(num_class, dtype='uint8')[y]
        return torch.LongTensor(to_cat)
    
    @staticmethod
    def collate():
        pass