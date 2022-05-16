import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy import sparse


class ClientsDataset:
    def __init__(self, data_path, neg_num):
        self.train_data = pd.read_csv(data_path, header=None,
                                      names=['user_id', 'item_id', 'timestamp'],
                                      dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})
        self.neg_num = neg_num
        self.train_data.sort_values(by=['user_id', 'timestamp'], axis='index', ascending=True, inplace=True,
                                    kind='mergesort')
        self.train_groupby = self.train_data.groupby('user_id')
        maxlen = self.train_groupby.count()['item_id'].max()
        self.user_set = self.train_data.user_id.unique()
        self.item_set = self.train_data.item_id.unique()
        self.user_maxid = self.train_data.user_id.unique().max().item()
        self.item_maxid = self.train_data.item_id.unique().max().item()
        self.train_seq = {}
        self.valid_seq = {}
        self.seq = {}
        self.seq_len = {}
        for u in self.user_set:
            items = self.train_groupby.get_group(u).item_id.to_numpy()
            train_s = np.pad(items[:-1], (0, maxlen + 1 - len(items)))
            valid_s = np.pad(items[1:], (0, maxlen + 1 - len(items)))
            self.seq_len[u] = len(items) - 1
            self.seq[u] = items
            self.valid_seq[u] = valid_s
            self.train_seq[u] = train_s
        self.item_prob = np.ones([self.item_maxid + 1], dtype=np.float32)

    def set_popularity(self, popularity):
        self.item_prob = popularity.astype(np.float32)

    def __len__(self):
        return len(self.user_set)

    def __getitem__(self, user_id):
        input_seq = self.train_seq[user_id]
        target_seq = self.valid_seq[user_id]
        input_len = self.seq_len[user_id]
        cand = np.setdiff1d(self.item_set, self.seq[user_id])
        prob = self.item_prob[cand]
        prob = prob / prob.sum()
        neg_seq = np.random.choice(cand, (input_len, self.neg_num), p=prob)
        neg_seq = np.pad(neg_seq, ((0, input_seq.shape[0] - input_len), (0, 0)))
        return input_seq, target_seq, input_len, neg_seq

    def get_maxid(self):
        return self.item_maxid

    def get_user_set(self):
        return self.user_set


class ClientsSampler(Dataset):
    def __init__(self, user_set):
        super().__init__()
        self.user_set = user_set

    def __len__(self):
        return len(self.user_set)

    def __getitem__(self, idx):
        return self.user_set[idx]


class TestDataset(Dataset):
    def __init__(self, data_path, target_path):
        self.train_data = pd.read_csv(data_path, header=None,
                                      names=['user_id', 'item_id', 'timestamp'],
                                      dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})
        self.target_data = pd.read_csv(target_path, header=None,
                                       names=['user_id', 'item_id', 'timestamp'],
                                       dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32})
        self.train_data.sort_values(by=['user_id', 'timestamp'], axis='index', ascending=True, inplace=True,
                                    kind='mergesort')
        self.train_groupby = self.train_data.groupby('user_id')
        maxlen = self.train_groupby.count()['item_id'].max()
        self.user_set = self.train_data.user_id.unique()
        self.item_set = self.train_data.item_id.unique()
        self.user_maxid = np.max(self.train_data.user_id.unique())
        self.item_maxid = np.max(self.train_data.item_id.unique())
        train_matrix = sparse.csr_matrix(
            (np.ones_like(self.train_data['user_id']), (self.train_data['user_id'], self.train_data['item_id'])),
            dtype='float64',
            shape=(self.user_maxid + 1, self.item_maxid + 1))
        self.train_matrix = train_matrix.toarray()
        target_matrix = sparse.csr_matrix(
            (np.ones_like(self.target_data['user_id']), (self.target_data['user_id'], self.target_data['item_id'])),
            dtype='float64',
            shape=(self.user_maxid + 1, self.item_maxid + 1))
        self.target_matrix = target_matrix.toarray()
        self.seq = {}
        self.seq_len = {}
        for u in self.user_set:
            items = self.train_groupby.get_group(u).item_id.to_numpy()
            seq = np.pad(items, (0, maxlen - len(items)))
            self.seq_len[u] = len(items)
            self.seq[u] = seq

    def __len__(self):
        return len(self.user_set)

    def __getitem__(self, idx):
        user_id = self.user_set[idx]
        input_seq = self.seq[user_id]
        input_len = self.seq_len[user_id]
        train_vec = self.train_matrix[user_id]
        target_vec = self.target_matrix[user_id]
        return input_seq, input_len, train_vec, target_vec
