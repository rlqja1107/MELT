from torch.utils.data import Dataset
import numpy as np
import torch
import random

class TrainData_FMLP(Dataset):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10):
        self.user_train = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.filter_user = self.filter_train_user()
        self.user_seq = []
        self.user_idx = []
        for u in self.filter_user:
            seq = self.user_train[u]
            input_ids = seq[-maxlen:]
            for i in range(len(input_ids)):
                if i == 0 :continue
                self.user_seq.append(input_ids[:i+1])
                self.user_idx.append(u)

    def filter_train_user(self):
        filter_user = []
        for k, v in self.user_train.items():
            if len(v) > 1:
                filter_user.append(k)
        return filter_user
                

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        items = self.user_seq[idx]
        user = self.user_idx[idx]
        input_ids = items[:-1]
        answer = items[-1]
        seq_set = set(items)
        pad_len = self.maxlen - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.maxlen:]
        assert len(input_ids) == self.maxlen
        neg = random.randint(1, self.itemnum)
        while neg in seq_set:
            neg = random.randint(1, self.itemnum)  
        cur_tensors = (
            torch.tensor(user, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )

        return cur_tensors # _, seq, pos, neg
    

class ValidData_FMLP(Dataset):
    def __init__(self, args, item_context, train_data, test_data, valid_data, neg_sampler):
        self.args = args
        self.test_user = np.array(list(dict((filter(lambda e:len(e[1])>0, test_data.items()))).keys()))
        self.user_seq = []
        self.user_idx = dict()
        self.user_idx = []
        for u in self.test_user:
            seq = train_data[u] + valid_data[u]
            self.user_seq.append(seq)
            self.user_idx.append(u)
            # self.user_idx[tuple(seq)] = u
        self.train_data = train_data
        self.inference_negative_sampler = neg_sampler
    
        
    def __len__(self):
        return len(self.test_user)

    def __getitem__(self, index):
        items = self.user_seq[index]
        # user = self.user_idx[items]
        user = self.user_idx[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)

        pad_len = self.args.maxlen - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.args.maxlen:]
        assert len(input_ids) == self.args.maxlen

        test_samples = self.inference_negative_sampler(user, is_valid='valid')
        cur_tensors = (
            torch.tensor(user, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            #torch.tensor(attribute, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(test_samples, dtype=torch.long),
        )
        return cur_tensors


class TestData_FMLP(Dataset):
    def __init__(self, args, item_context, train_data, test_data, valid_data, neg_sampler):
        self.args = args
        self.test_user = np.array(list(dict((filter(lambda e:len(e[1])>0, test_data.items()))).keys()))
        self.user_seq = []
        self.user_idx = dict()
        self.user_idx = []
        for u in self.test_user:
            seq = train_data[u] + valid_data[u] + test_data[u]
            self.user_seq.append(seq)
            # self.user_idx[tuple(seq)] = u
            self.user_idx.append(u)
        self.train_data = train_data
        self.inference_negative_sampler = neg_sampler
    
        
    def __len__(self):
        return len(self.test_user)

    def __getitem__(self, index):
        items = self.user_seq[index]
        user = self.user_idx[index]
        # user = self.user_idx[tuple(items)]
        
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)

        pad_len = self.args.maxlen - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.args.maxlen:]
        assert len(input_ids) == self.args.maxlen

        test_samples = self.inference_negative_sampler(user, is_valid='test')
        cur_tensors = (
            torch.tensor(user, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            #torch.tensor(attribute, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(test_samples, dtype=torch.long),
        )
        return cur_tensors