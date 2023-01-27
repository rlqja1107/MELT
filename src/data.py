from torch.utils.data import Dataset
import numpy as np
import torch
import random

class TrainData(Dataset):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10):
        self.user_train = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.filter_user = self.filter_train_user()

    def filter_train_user(self):
        filter_user = []
        for k, v in self.user_train.items():
            if len(v) > 1:
                filter_user.append(k)
        return filter_user
                

    def __len__(self):
        return len(self.filter_user)

    def __getitem__(self, idx):
        user = self.filter_user[idx]
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1

        ts = set(self.user_train[user])
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg_item = np.random.randint(1, self.itemnum+1)
                while neg_item in ts:
                    neg_item = np.random.randint(1, self.itemnum+1)
                neg[idx] = neg_item
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)



class TestData(Dataset):
    def __init__(self, args, item_context, train_data, test_data, valid_data, neg_sampler):
        self.args = args
        self.item_context = item_context
        self.test_user = np.array(list(dict((filter(lambda e:len(e[1])>0, test_data.items()))).keys()))
        self.train_data = train_data
        self.inference_negative_sampler = neg_sampler
        self.valid_data = valid_data
        self.test_data = test_data
        
    def __len__(self):
        return len(self.test_user)

    def __getitem__(self, index):
        test_user = self.test_user[index]
        seq = np.zeros(self.args.maxlen, dtype=np.int32)
        idx = self.args.maxlen - 1

        seq[idx] = self.valid_data[test_user][0]
        idx -= 1

        for i in reversed(self.train_data[test_user]):
            seq[idx] = i
            idx -= 1
            if idx==-1:break

        item_idx = self.inference_negative_sampler(test_user, is_valid='test')
        return test_user, seq, torch.LongTensor(item_idx), self.test_data[test_user][0]
    


class ValidData(Dataset):
    def __init__(self, args, item_context, train_data, test_data, valid_data, neg_sampler):
        self.args = args
        self.item_context = item_context
        self.test_user = np.array(list(dict((filter(lambda e:len(e[1])>0, test_data.items()))).keys()))
        self.train_data = train_data
        self.inference_negative_sampler = neg_sampler
        self.valid_data = valid_data
        self.test_data = test_data
        
    def __len__(self):
        return len(self.test_user)

    def __getitem__(self, index):
        test_user = self.test_user[index]
        seq = np.zeros(self.args.maxlen, dtype=np.int32)
        idx = self.args.maxlen - 1

        for i in reversed(self.train_data[test_user]):
            seq[idx] = i
            idx -= 1
            if idx==-1:break

        item_idx = self.inference_negative_sampler(test_user, is_valid='valid')
        return test_user, seq, torch.LongTensor(item_idx), self.valid_data[test_user][0]