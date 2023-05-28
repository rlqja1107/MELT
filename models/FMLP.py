import os
import sys
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from src.utils import setupt_logger, set_random_seeds, Checker
from src.sampler import NegativeSampler
from embedder import embedder
import time
from src.data_FMLP import TrainData_FMLP, ValidData_FMLP, TestData_FMLP
import torch.utils.data as data_utils
from tqdm import tqdm
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

class Trainer(embedder):

    def __init__(self, args, logger=None):
        if logger is None:
            self.logger = setupt_logger(args, f'log/{args.model}/{args.dataset}', name = args.model, filename = "log.txt")
        else:
            self.logger = logger
        self.logger.info(args)
        self.args = args
        embedder.__init__(self, args, self.logger)
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        self.split_head_tail()


    def train(self):
        set_random_seeds(self.args.seed)
        self.model = FMLPRec(self.args, self.item_num, self.device).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        # Build train, valid, test datasets
        train_dataset = TrainData_FMLP(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size, maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        valid_dataset = ValidData_FMLP(self.args, None, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.valid_loader = data_utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        test_dataset = TestData_FMLP(self.args, None, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        self.optimzer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
        self.validcheck = Checker(self.logger)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(self.args.e_max):
            start = time.time()
            rec_loss = 0.0
            self.model.train()
            for _, batch in enumerate(train_loader):
                batch = tuple(t.to(self.device) for t in batch)
                u, seq, pos, neg = batch
                sequence_output = self.model(seq)

                loss = self.cross_entropy(sequence_output, pos, neg)
                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()
                rec_loss += loss.item()
                
            end = time.time()
            print(f'Epoch: {epoch}, Time: {end-start:.4f}, Evaluating: Dataset({self.args.dataset}), Model: ({self.args.model}), GPU: {self.args.gpu}')
            
            if epoch % 2 == 0:
                with torch.no_grad():
                    self.model.eval()
                    result_valid = self.evaluate(self.model, k=10, is_valid="valid")
                    print(f"Epoch:{epoch}, Valid (NDCG@10: {result_valid['Overall']['NDCG']:.4f}), HR@10: {result_valid['Overall']['HIT']:.4f})")
                    best_valid = self.validcheck(result_valid, epoch, self.model, f'{self.args.model}_{self.args.dataset}.pth')
                    if best_valid:
                        result_10 = self.evaluate(self.model, k=10, is_valid="test")

        folder = f"save_model/{self.args.dataset}"
        os.makedirs(folder, exist_ok=True)
        torch.save(self.validcheck.best_model.state_dict(), os.path.join(folder, self.validcheck.best_name))
        
        with torch.no_grad():
            self.validcheck.best_model.eval()
            result_5 = self.evaluate(self.validcheck.best_model, k=5, is_valid='test')
            result_10 = self.evaluate(self.validcheck.best_model, k=10, is_valid='test')
            self.validcheck.refine_test_result(result_5, result_10)
            self.validcheck.print_result()


    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        seq_emb = seq_out[:, -1, :]
        pos_logits = torch.sum(pos_emb * seq_emb, -1)
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )
        return loss

    
    def test(self):
        set_random_seeds(self.args.seed)
        self.model = FMLPRec(self.args, self.item_num, self.device).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        test_dataset = TestData_FMLP(self.args, None, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        self.printer = Checker(self.logger)
        
        # Take the pre-trained model
        model_path = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model.eval()
        
        # Evaluate
        with torch.no_grad():
            result_5 = self.evaluate(self.model, k=5, is_valid='test')
            result_10 = self.evaluate(self.model, k=10, is_valid='test')
            self.printer.refine_test_result(result_5, result_10)
            self.printer.print_result()
            
        
    def evaluate(self, model, k=10, is_valid='test'):
        """
        To evaluate test dataset
        """
        HT = 0.0
        NDCG = 0.0
        
        TAIL_USER_NDCG = 0.0
        HEAD_USER_NDCG = 0.0
        TAIL_ITEM_NDCG = 0.0
        HEAD_ITEM_NDCG = 0.0
        
        TAIL_USER_HIT = 0.0
        HEAD_USER_HIT = 0.0
        TAIL_ITEM_HIT = 0.0
        HEAD_ITEM_HIT = 0.0
        
        n_all_user = 0.0
        head_user = 0.0
        tail_user = 0.0
        head_item = 0.0
        tail_item = 0.0

        loader = self.test_loader if is_valid == 'test' else self.valid_loader
        
        for _, batch in enumerate(loader):
            batch = tuple(t.to(self.device) for t in batch)
            u, seq, test_idx, item_idx = batch
            u = u.cpu(); test_idx = test_idx.cpu()
            predictions = model(seq)
            predictions = predictions[:, -1, :]
            item_embs = model.item_embeddings(item_idx) 
            
            test_logits = torch.bmm(item_embs, predictions.unsqueeze(-1)).squeeze(-1)
            test_logits = -test_logits
            rank = test_logits.argsort(1).argsort(1)[:,0].cpu().numpy()

            hit_user = rank < k
            ndcg = 1 / np.log2(rank + 2)

            n_all_user += len(test_logits)
            u_head = (self.u_head_set[None, ...] == u.numpy()[...,None]).nonzero()[0]
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[...,None]).nonzero()[0]
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)
            
            head_user += len(u_head)
            tail_user += len(u_tail)
            head_item += len(i_head)
            tail_item += len(i_tail)
            
            HT += np.sum(hit_user).item()
            HEAD_USER_HIT += sum(hit_user[u_head])
            TAIL_USER_HIT += sum(hit_user[u_tail])
            HEAD_ITEM_HIT += sum(hit_user[i_head])
            TAIL_ITEM_HIT += sum(hit_user[i_tail])
            
            NDCG += np.sum(1 / np.log2(rank[hit_user] + 2)).item()
            HEAD_ITEM_NDCG += sum(ndcg[i_head[hit_user[i_head]]])
            TAIL_ITEM_NDCG += sum(ndcg[i_tail[hit_user[i_tail]]])
            HEAD_USER_NDCG += sum(ndcg[u_head[hit_user[u_head]]])
            TAIL_USER_NDCG += sum(ndcg[u_tail[hit_user[u_tail]]])
            
        result = {'Overall': {'NDCG': NDCG / n_all_user, 'HIT': HT / n_all_user}, 
                'Head_User': {'NDCG': HEAD_USER_NDCG / head_user, 'HIT': HEAD_USER_HIT / head_user},
                'Tail_User': {'NDCG': TAIL_USER_NDCG / tail_user, 'HIT': TAIL_USER_HIT / tail_user},
                'Head_Item': {'NDCG': HEAD_ITEM_NDCG / head_item, 'HIT': HEAD_ITEM_HIT / head_item},
                'Tail_Item': {'NDCG': TAIL_ITEM_NDCG / tail_item, 'HIT': TAIL_ITEM_HIT / tail_item},
                }

        return result



class FMLPRec(torch.nn.Module):
    """
    Revision of SASRec model
    """
    def __init__(self, args, item_num, device):
        super(FMLPRec, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(item_num + 1 , args.hidden_units, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.item_encoder = Encoder(args)
        self.apply(self.init_weights)
        self.device = device


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb
  

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        sequence_output = item_encoded_layers[-1]

        return sequence_output



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_blocks)]) # 2

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.filterlayer(hidden_states)

        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.maxlen//2 + 1, args.hidden_units, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.dropout_rate)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)


    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}
    
class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_units, args.hidden_units * 4)
        self.intermediate_act_fn = gelu

        self.dense_2 = nn.Linear(4 * args.hidden_units, args.hidden_units)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
