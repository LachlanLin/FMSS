import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path, sep='\\s+', header=None, names=['uid', 'sid', 'rating', 'time'],
                                engine='python')
        self.data.drop(['time'], axis=1, inplace=True)
        self.data['uid'] = self.data['uid'] - 1
        self.data['sid'] = self.data['sid'] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        uid = record['uid']
        iid = record['sid']
        rui = record['rating']
        return torch.tensor(uid), torch.tensor(iid), torch.tensor(rui, dtype=torch.float32)


class ClientsSampler(Dataset):
    def __init__(self, n):
        super().__init__()
        self.users_seq = np.arange(n)

    def __len__(self):
        return len(self.users_seq)

    def __getitem__(self, idx):
        return self.users_seq[idx]


class ClientsDataset:
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep='\\s+', header=None, names=['uid', 'sid', 'rating', 'time'],
                           engine='python')
        data.drop(['time'], axis=1, inplace=True)
        data['uid'] = data['uid'] - 1
        data['sid'] = data['sid'] - 1
        self.dataByUser = data.groupby('uid')

    def __len__(self):
        return len(self.dataByUser)

    def __getitem__(self, idx):
        records = self.dataByUser.get_group(idx)
        iid = records['sid'].to_numpy()
        rui = records['rating'].to_numpy()
        return torch.tensor(iid), torch.tensor(rui, dtype=torch.float32)
