import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from embedder import embedder
from src.sampler import NegativeSampler
from models.FMLP import Trainer as Trainer_FMLP
from src.utils import setupt_logger, set_random_seeds, Checker
from src.data_FMLP import TrainData_FMLP, ValidData_FMLP, TestData_FMLP


class Trainer(embedder):

    def __init__(self, args):
        self.logger = setupt_logger(args, f'log/{args.model}/{args.dataset}', name = args.model, filename = "log.txt")
        embedder.__init__(self, args, self.logger)
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"
        self.args = args
        torch.cuda.set_device(self.device)
        self.split_head_tail()
        self.save_user_item_context()

    def train(self):
        set_random_seeds(self.args.seed)
        u_L_max = self.args.maxlen
        i_L_max = self.item_threshold + self.args.maxlen
        self.model = MELT(self.args, self.logger, self.item_num, self.device, u_L_max, i_L_max).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        # Build train, valid, test datasets
        train_dataset = TrainData_FMLP(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size, maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        valid_dataset = ValidData_FMLP(self.args, None, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.valid_loader = data_utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        test_dataset = TestData_FMLP(self.args, None, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
        adam_optimizer= torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0)
        
        # For selecting the best model
        self.validcheck = Checker(self.logger)
        
        # For updating the tail item embedding
        i_tail_set = data_utils.TensorDataset(torch.LongTensor(list(self.i_tail_set)))
        i_tail_loader = data_utils.DataLoader(i_tail_set, self.args.batch_size * 2, shuffle=False, drop_last=False)
        
        # DataLoader for bilateral branches
        i_h_data_loader = data_utils.DataLoader(self.i_head_set, int(self.args.branch_batch_size), shuffle=True, drop_last=False)
        u_h_loader = data_utils.DataLoader(self.u_head_set, int(self.args.branch_batch_size), shuffle=True, drop_last=False)
        
        
        for epoch in range(self.args.e_max):
            training_loss = 0.0
            self.model.eval()
            
            with torch.no_grad():
                # Knowledge transfer from item branch to user branch
                self.model.update_item_tail_embedding(i_tail_loader, self.item_context)
                
            self.model.train()
            for _, ((batch), (u_idx), (i_idx)) in enumerate(zip(train_loader, u_h_loader, i_h_data_loader)):
                batch = tuple(t.to(self.device) for t in batch)
                u_idx = u_idx.numpy()
                i_idx = i_idx.numpy()
                _, seq, pos, neg = batch
                
                # Sequence Encoder
                sequence_output = self.model(seq)
                
                # Next item Prediction Loss
                prediction_loss = self.cross_entropy(sequence_output, pos, neg)
                
                # User branch loss (L_u)
                user_loss = self.model.user_branch(u_idx, self.model, self.user_context, self.user_threshold, epoch)
                
                # Item branch loss (L_i)
                item_loss = self.model.item_branch(self.item_context, i_idx, self.model, self.item_threshold, self.model.user_branch.W_U, epoch=epoch, n_item_context=self.n_item_context)
                
                # Final Loss (L_{final})
                loss = user_loss * self.args.lamb_u + item_loss * self.args.lamb_i + prediction_loss

                adam_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()
                training_loss += loss.item()
                
            if epoch % 2 == 0:
                with torch.no_grad():
                    self.model.eval()
                    result_valid = self.evaluate(self.model, k=10, is_valid="valid")
                    best_valid = self.validcheck(result_valid, epoch, self.model, f'{self.args.model}_{self.args.dataset}.pth')
           
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}), Loss: ({training_loss:.3f}), GPU: {self.args.gpu}')
            
        folder = f"save_model/{self.args.dataset}"
        os.makedirs(folder, exist_ok=True)
        torch.save(self.validcheck.best_model.state_dict(), os.path.join(folder, self.validcheck.best_name))
        
        with torch.no_grad():
            self.validcheck.best_model.eval()
            result_5 = self.evaluate(self.validcheck.best_model, k=5, is_valid="test")
            result_10 = self.evaluate(self.validcheck.best_model, k=10, is_valid="test")
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
        u_L_max = self.args.maxlen
        i_L_max = self.item_threshold + self.args.maxlen
        self.model = MELT(self.args, self.logger, self.item_num, self.device, u_L_max, i_L_max, True).to(self.device)
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


    def evaluate(self, model, k=10, is_valid="test"):
        """
        Evaluation on validation or test set
        """
        HIT = 0.0  # Overall Hit
        NDCG = 0.0 # Overall NDCG
        TAIL_USER_NDCG = 0.0
        HEAD_USER_NDCG = 0.0
        TAIL_ITEM_NDCG = 0.0
        HEAD_ITEM_NDCG = 0.0
        
        TAIL_USER_HIT = 0.0
        HEAD_USER_HIT = 0.0
        TAIL_ITEM_HIT = 0.0
        HEAD_ITEM_HIT = 0.0
        
        n_all_user = 0.0
        n_head_user = 0.0
        n_tail_user = 0.0
        n_head_item = 0.0
        n_tail_item = 0.0

        loader = self.test_loader if is_valid == "test" else self.valid_loader
        
        for _, (u, seq, test_idx, item_idx) in enumerate(loader):
            u_head = (self.u_head_set[None, ...] == u.numpy()[...,None]).nonzero()[0]        # Index of head users
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)                                 # Index of tail users
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[...,None]).nonzero()[0] # Index of head items
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)                                 # Index of tail items
            
            predictions = model(seq.to(self.device))
            predictions = predictions[:, -1, :]
            
            # Knowledge transfer to tail users
            predictions[u_tail] += model.user_branch.W_U(predictions)[u_tail] 
            item_embs = model.item_embeddings(item_idx.to(self.device)) 

            test_logits = torch.bmm(item_embs, predictions.unsqueeze(-1)).squeeze(-1)
            test_logits = -test_logits
            rank = test_logits.argsort(1).argsort(1)[:,0].cpu().numpy()
            n_all_user += len(test_logits)

            hit_user = rank < k
            ndcg = 1 / np.log2(rank + 2)

            n_head_user += len(u_head)
            n_tail_user += len(u_tail)
            n_head_item += len(i_head)
            n_tail_item += len(i_tail)
            
            HIT += np.sum(hit_user).item()
            HEAD_USER_HIT += sum(hit_user[u_head])
            TAIL_USER_HIT += sum(hit_user[u_tail])
            HEAD_ITEM_HIT += sum(hit_user[i_head])
            TAIL_ITEM_HIT += sum(hit_user[i_tail])
            
            NDCG += np.sum(1 / np.log2(rank[hit_user] + 2)).item()
            HEAD_ITEM_NDCG += sum(ndcg[i_head[hit_user[i_head]]])
            TAIL_ITEM_NDCG += sum(ndcg[i_tail[hit_user[i_tail]]])
            HEAD_USER_NDCG += sum(ndcg[u_head[hit_user[u_head]]])
            TAIL_USER_NDCG += sum(ndcg[u_tail[hit_user[u_tail]]])
            torch.cuda.empty_cache()
        result = {'Overall': {'NDCG': NDCG / n_all_user, 'HIT': HIT / n_all_user}, 
                'Head_User': {'NDCG': HEAD_USER_NDCG / n_head_user, 'HIT': HEAD_USER_HIT / n_head_user},
                'Tail_User': {'NDCG': TAIL_USER_NDCG / n_tail_user, 'HIT': TAIL_USER_HIT / n_tail_user},
                'Head_Item': {'NDCG': HEAD_ITEM_NDCG / n_head_item, 'HIT': HEAD_ITEM_HIT / n_head_item},
                'Tail_Item': {'NDCG': TAIL_ITEM_NDCG / n_tail_item, 'HIT': TAIL_ITEM_HIT / n_tail_item}
                }

        return result


class USERBRANCH(torch.nn.Module):
    def __init__(self, args, device, u_L_max):
        """
        User branch: Enhance the tail user representation
        """
        super(USERBRANCH, self).__init__()
        self.args = args
        self.W_U = torch.nn.Linear(self.args.hidden_units, self.args.hidden_units)
        torch.nn.init.xavier_normal_(self.W_U.weight.data)
        # self.transfer =transfer
        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.u_L_max = u_L_max
        self.pi = np.pi
        
        
    def forward(self, u_head_idx, model, user_context, user_thres, epoch):
        full_seq = np.zeros(([len(u_head_idx), self.args.maxlen]), dtype=np.int32)
        w_u_list = []
        
        for i, u_h in enumerate(u_head_idx):
            full_seq[i] = user_context[u_h]
            seq_length = user_context[u_h].nonzero()[0].shape[0]
            
            # Calculate the loss coefficient
            w_u = (self.pi/2)*(epoch/self.args.e_max)+(self.pi/(2*(self.u_L_max-user_thres-1)))*(seq_length-user_thres-1)
            w_u = np.abs(np.sin(w_u))
            w_u_list.append(w_u)
            
        # Representations of full sequence
        full_seq_repre = model(torch.LongTensor(full_seq).to(self.device))[:, -1,:]
        few_seq = np.zeros([len(u_head_idx), self.args.maxlen], dtype=np.int32)

        R = np.random.randint(1, user_thres, len(full_seq))
        for i, l in enumerate(R):
            few_seq[i, -l:] = full_seq[i, -l:]
            
        # Representations of recent interactions 
        few_seq_repre = model(torch.LongTensor(few_seq).to(self.device))[:, -1, :]
        w_u_list = torch.FloatTensor(w_u_list).view(-1,1).to(self.device)
        
        # Curriculum Learning by user
        loss = (w_u_list*((self.W_U(few_seq_repre) - full_seq_repre) ** 2)).mean()
        return loss
    

class ITEMBRANCH(torch.nn.Module):
    def __init__(self, args, device, i_L_max):
        """
        Item branch: Enhance the tail item representation
        """
        super(ITEMBRANCH, self).__init__()
        self.args = args
        self.device = device
        self.W_I = torch.nn.Linear(args.hidden_units, args.hidden_units)
        torch.nn.init.xavier_normal_(self.W_I.weight.data)
        self.criterion = torch.nn.MSELoss()
        self.i_L_max = i_L_max
        self.pi = np.pi
        
        
    def forward(self, item_context, i_head_idx, model, item_thres, W_U, epoch=None, n_item_context=None):
        target_embed = []
        subseq_set = []
        subseq_set_idx = [0]
        idx = 0
        w_i_list = []
        
        for h_i in i_head_idx:
            item_context_list = np.vstack(item_context[h_i])
            n_context = min(self.i_L_max, n_item_context[h_i])
            
            #  Calculate the loss coefficient      
            w_i = (self.pi/2)*(epoch/self.args.e_max)+(self.pi/100)*(n_context-(item_thres+1))
            w_i = np.abs(np.sin(w_i))
            w_i_list.append(w_i)           
            len_context = len(item_context[h_i])
            
            # Set upper bound of user seq.
            thres = min(len_context, item_thres)
            n_few_inter = np.random.randint(1, thres+1)
            
            # Randomly sample the contexts
            K = np.random.choice(range(len(item_context_list)), int(n_few_inter), replace=False)
            idx += len(K)
            
            subseq_set.append(item_context_list[K])
            target_embed.append(h_i)
            subseq_set_idx.append(idx)
            
        subseq_set = torch.LongTensor(np.vstack(subseq_set)).to(self.device)
        
        # Encode the subsequence set
        sequence_emb = model(subseq_set)[:,-1,:]
        
        # Knowledge transfer from user branch to item branch
        sequence_emb = sequence_emb + W_U(sequence_emb)
        
        # Contextualized representations
        subseq_set = []
        for i, h_i in enumerate(i_head_idx):
            encode_average_embed = sequence_emb[subseq_set_idx[i]:subseq_set_idx[i+1]]
            subseq_set.append(encode_average_embed.mean(0))
            
        few_subseq_embed = torch.stack(subseq_set)
        target_embed = torch.LongTensor(target_embed).to(self.device)
        w_i_list = torch.FloatTensor(w_i_list).view(-1,1).to(self.device)
        
        # Curriculum Learning by item
        loss = (w_i_list*((self.W_I(few_subseq_embed) - model.item_embeddings(target_embed))) ** 2).mean()
        return loss 



class MELT(torch.nn.Module):
    """
    Revision of SASRec model
    """
    def __init__(self, args, logger, item_num, device, u_L_max, i_L_max, test=False):
        super(MELT, self).__init__()
        self.args = args
        self.logger = logger
        self.test = test
        self.item_embeddings = nn.Embedding(item_num + 1 , args.hidden_units, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.item_encoder = Encoder(args)
        self.apply(self.init_weights)
        self.device = device
        if not self.test:
            self.load_pretrained_model()
        self.user_branch = USERBRANCH(self.args, self.device, u_L_max).to(self.device)
        self.item_branch = ITEMBRANCH(self.args, self.device, i_L_max).to(self.device)
        
        
    def load_pretrained_model(self):
        try:
            model_path = f"save_model/{self.args.dataset}/FMLP_{self.args.dataset}.pth"
            self.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))        
        except:
            self.logger.info("No trained model in path")
            self.logger.info("Train the FMLP model")
            fmlp = Trainer_FMLP(self.args, self.logger)
            fmlp.train()
            model_path = f"save_model/{self.args.dataset}/FMLP_{self.args.dataset}.pth"
            self.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

         

    def update_item_tail_embedding(self, i_tail_loader, item_context):
        """
        Update the tail item representations
        It enhances the tail item representation in users' sequence and item pools at once.
        """
        for _, i_t in enumerate(i_tail_loader):
            tail_idx = []
            collect_context = []
            i_tail_idx = [0]
            cumulative_idx = 0
            for i in i_t[0]:
                i = i.item()
                if len(item_context[i]) >=1:
                    stack_context = np.vstack(item_context[i])
                    cumulative_idx += len(stack_context)
                    i_tail_idx.append(cumulative_idx)
                    
                    collect_context.extend(stack_context)
                    tail_idx.append(i)
            collect_context = torch.LongTensor(np.vstack(collect_context)).to(self.device)
            group_context_embed = self.forward(collect_context)[:,-1,:]
            i_tail_average_emb = []
            idx = 0 
            for i in i_t[0]:
                i = i.item()
                if len(item_context[i]) >=1:
                    i_encode_emb = group_context_embed[i_tail_idx[idx]:i_tail_idx[idx+1]]
                    i_tail_average_emb.append(i_encode_emb.mean(0))
                    idx+=1
            group_fully_context_embed = torch.stack(i_tail_average_emb)
            i_tail_estimate_embed = self.item_branch.W_I(group_fully_context_embed)
            
            tail_idx = torch.LongTensor(tail_idx).to(self.device)
            self.item_embeddings.weight.data[tail_idx] = i_tail_estimate_embed


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
        seq_length = sequence.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb


    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.item_embeddings(pos_ids)
        neg_emb = self.item_embeddings(neg_ids)

        seq_emb = seq_out[:, -1, :] # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )

        return loss
  


    def forward(self, input_ids):
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                output_all_encoded_layers=True
                                                )
        sequence_output = item_encoded_layers[-1]
        return sequence_output



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_blocks)]) # 2

        
    def forward(self, hidden_states, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
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


    def forward(self, hidden_states):
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
