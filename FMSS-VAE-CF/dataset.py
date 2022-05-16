import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, train_path, test_path, n, m):
        super().__init__()
        train_data = pd.read_csv(train_path, sep='\\s+', header=None, names=['uid', 'sid'], engine='python')
        train_data['uid'] = train_data['uid'] - 1
        train_data['sid'] = train_data['sid'] - 1
        rows_tr, cols_tr = train_data['uid'], train_data['sid']
        data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                    dtype='float64',
                                    shape=(n, m))
        self.train_dataset = torch.FloatTensor(data_tr.toarray())
        test_data = pd.read_csv(test_path, sep='\\s+', header=None, names=['uid', 'sid'], engine='python')
        test_data['uid'] = test_data['uid'] - 1
        test_data['sid'] = test_data['sid'] - 1
        rows_te, cols_te = test_data['uid'], test_data['sid']
        data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                    dtype='float64',
                                    shape=(n, m))
        self.test_dataset = torch.FloatTensor(data_te.toarray())

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        train_record = self.train_dataset[idx]
        test_record = self.test_dataset[idx]
        return train_record, test_record


class ClientsSampler(Dataset):
    def __init__(self, n):
        super().__init__()
        self.users_seq = np.arange(n)

    def __len__(self):
        return len(self.users_seq)

    def __getitem__(self, idx):
        return self.users_seq[idx]


class ClientsDataset:
    def __init__(self, data_path, n, m):
        data = pd.read_csv(data_path, sep='\\s+', header=None, names=['uid', 'sid'], engine='python')
        data['uid'] = data['uid'] - 1
        data['sid'] = data['sid'] - 1
        rows, cols = data['uid'], data['sid']
        clients_data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)),
                                         dtype='float64',
                                         shape=(n, m))
        self.clients_data = torch.FloatTensor(clients_data.toarray())

    def __len__(self):
        return len(self.clients_data)

    def __getitem__(self, idx):
        return self.clients_data[idx]
