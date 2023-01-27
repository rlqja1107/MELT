from collections import Counter
import numpy as np


class NegativeSampler(object):
    def __init__(self, args, dataset):
        """
        For inference stage, sample the negative items not interacted with user
        dataset : [user_train, user_valid, user_test, user_time_diff, usernum, itemnum]
        """
        self.args = args
        self.dataset= dataset
        self.usernum = dataset[4] # User num
        self.itemnum = dataset[5] # Item num


        self.negative_samples = self.get_random_negative('test')     
        self.negative_samples_valid = self.get_random_negative('valid')
        
    def __call__(self, user, is_valid):
        if is_valid=='test':
            return self.negative_samples[user] 
        else:
            return self.negative_samples_valid[user]

    def get_random_negative(self, valid_or_test):
        """
        For testing, bring the random items not included in user sequence
        """
        np.random.seed(self.args.seed)
        negative_samples = {}
        for user in np.arange(1, self.usernum+1):
            if len(self.dataset[2][user]) < 1:
                continue
            seen = set(self.dataset[0][user])

            seen.add(0)
            samples = []
            if valid_or_test == 'test':
                seen.add(self.dataset[1][user][0])
                samples.append(self.dataset[2][user][0]) # Put gt item
            else:
                samples.append(self.dataset[1][user][0]) # Put gt item
            for _ in range(self.args.n_negative_samples):
                t = np.random.randint(1, self.itemnum + 1)
                while t in seen: t = np.random.randint(1, self.itemnum + 1)
                samples.append(t)
                seen.add(t)
            negative_samples[user] = samples
        return negative_samples


