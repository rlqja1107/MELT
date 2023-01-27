import os
import sys
import copy
import torch
import random
import numpy as np
from copy import deepcopy
import logging


def set_random_seeds(seed):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


def setupt_logger(args, save_dir, name, filename = 'log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(4)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("================================================")
    return logger


class Checker(object):
    def __init__(self, logger):
        self.logger = logger
        self.cur_tolerate = 0
        self.best_val = {'Overall':{'HIT':-1}}
        self.best_result_5 = None
        self.best_result_10 = None
        self.best_epoch = 0
        self.best_model = None
        self.best_name = None
        # Pretrain
        self.min_loss = 10000000.0


    def print_result(self):
        self.logger.info("==========================================================")
        self.logger.info(f"Validation: HIT@10: {self.best_val['Overall']['HIT']:.4f}")
        self.logger.info(f"Overall Result: NDCG@10: {self.best_result_10['Overall']['NDCG']:.4f}, HIT@10: {self.best_result_10['Overall']['HIT']:.4f}")
        self.logger.info(f"Head User - NDCG({self.best_result_10['Head_User']['NDCG']:.4f}), Hit({self.best_result_10['Head_User']['HIT']:.4f})")
        self.logger.info(f"Tail User - NDCG({self.best_result_10['Tail_User']['NDCG']:.4f}), Hit({self.best_result_10['Tail_User']['HIT']:.4f})")
        self.logger.info(f"Head Item - NDCG({self.best_result_10['Head_Item']['NDCG']:.4f}), Hit({self.best_result_10['Head_Item']['HIT']:.4f})")
        self.logger.info(f"Tail Item - NDCG({self.best_result_10['Tail_Item']['NDCG']:.4f}), Hit({self.best_result_10['Tail_Item']['HIT']:.4f})")


    def refine_test_result(self, result_5, result_10):
        self.best_result_5 = result_5
        self.best_result_10 = result_10
        
    def __call__(self, cur_valid, epoch, model, name):
        """
        For main training
        """
        if self.best_val['Overall']['HIT'] < cur_valid['Overall']['HIT']:
            self.best_val = cur_valid
            self.best_model = deepcopy(model)
            self.best_epoch = epoch
            self.best_name = name
            return True
        else:
            return False

