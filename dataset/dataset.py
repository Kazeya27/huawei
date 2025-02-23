# -*- coding: utf-8 -*-
"""
@author:
"""


import numpy as np
from torch.utils.data import Dataset


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class TimeSeries(Dataset):
    def __init__(self, data,
                 seq_len, pred_len, partition,
                 frac_test=0.2, frac_valid=0.2, frac_train=0.6):
        assert partition in ['train', 'valid', 'test', ]

        self.seq_len = seq_len
        self.pred_len = pred_len

        n_data = data.shape[0]
        n_test = int(n_data * frac_test)
        n_valid = int(n_data * frac_valid)
        data_usage = frac_test + frac_valid + frac_train
        if np.isclose(data_usage, 1.0):
            n_train = n_data - n_test - n_valid
            n_nonuse = 0
        else:
            n_train = int(n_data * frac_train)
            n_nonuse = n_data - n_test - n_valid - n_train

        train_start = max(0, n_nonuse - seq_len)
        train_end = n_nonuse + n_train
        data_mu = np.mean(data[train_start:train_end, :],
                          axis=0, keepdims=True)
        data_sigma = np.std(data[train_start:train_end, :],
                            axis=0, keepdims=True)
        data_sigma[0, data_sigma[0, :] < 1e-6] = 1

        self.scaler = StandardScaler(data_mu, data_sigma)

        if partition == 'train':
            segment_start = max(0, n_nonuse - seq_len)
            segment_end = n_nonuse + n_train
        elif partition == 'valid':
            segment_start = n_nonuse + n_train - seq_len
            segment_end = n_nonuse + n_train + n_valid
        elif partition == 'test':
            segment_start = n_nonuse + n_train + n_valid - seq_len
            segment_end = n_data

        self.partition = partition
        self.data = data[segment_start:segment_end, :]
        self.data = self.scaler.transform(self.data)
        self.num_nodes = self.data.shape[1]

        self.n_data = self.data.shape[0] - seq_len - pred_len + 1

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):
        seq_len = self.seq_len
        pred_len = self.pred_len

        x_start = index
        x_end = x_start + seq_len

        y_start = x_end
        y_end = y_start + pred_len

        data_x = self.data[x_start:x_end, :].T
        data_y = self.data[y_start:y_end, :].T
        return data_x, data_y

    def get_feature(self):
        return {"scaler": self.scaler, "num_nodes": self.num_nodes}


