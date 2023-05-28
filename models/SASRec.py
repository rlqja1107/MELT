import os
import numpy as np
import torch
from src.utils import setupt_logger, set_random_seeds, Checker
from src.sampler import NegativeSampler
from embedder import embedder
from src.data import ValidData, TestData, TrainData
import torch.utils.data as data_utils


class Trainer(embedder):

    def __init__(self, args, logger=None):
        self.args = args
        if logger is None:
            self.logger = setupt_logger(args, f'log/{args.model}/{args.dataset}', name = args.model, filename = f'log.txt')
        else:
            self.logger = logger
        embedder.__init__(self, args, self.logger)
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        self.split_head_tail()
        self.save_user_item_context()


    def train(self):
        """
        Train the model
        """
        set_random_seeds(self.args.seed)
        self.logger.info(f"============Start Training (SASRec)=======================")
        self.model = SASRec(self.args, self.item_num).to(self.device)
        self.init_param()

        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        # Build the train, valid, test datasets
        train_dataset = TrainData(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size, maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        valid_dataset = ValidData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.valid_loader = data_utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        self.validcheck = Checker(self.logger)
        
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        for epoch in range(1, self.args.e_max + 1):
            self.model.train()
            training_loss = 0.0
            for _, (u, seq, pos, neg) in enumerate(train_loader):
                adam_optimizer.zero_grad()
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = self.model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.device), torch.zeros(neg_logits.shape).to(self.device)
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                for param in self.model.item_emb.parameters(): loss += self.args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                training_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}), Model: ({self.args.model}), Training Loss: {training_loss:.3f}')
            
            if epoch % 5 == 0:
                self.model.eval()
                result_valid = self.evaluate(self.model, k=10, is_valid='valid')
                best_valid = self.validcheck(result_valid, epoch, self.model, f'{self.args.model}_{self.args.dataset}.pth')
        # Evaluation
        with torch.no_grad():
            self.validcheck.best_model.eval()
            result_5 = self.evaluate(self.validcheck.best_model, k=5, is_valid='test')
            result_10 = self.evaluate(self.validcheck.best_model, k=10, is_valid='test')
            self.validcheck.refine_test_result(result_5, result_10)
            self.validcheck.print_result()


        folder = f"save_model/{self.args.dataset}"
        os.makedirs(folder, exist_ok=True)
        torch.save(self.validcheck.best_model.state_dict(), os.path.join(folder, self.validcheck.best_name))
        self.validcheck.print_result()
            
           
    def  init_param(self):
        """
        Initialization of parameters
        """
        for _, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass


    def test(self):
        """
        Inference with trained model
        """
        set_random_seeds(self.args.seed)
        self.model = SASRec(self.args, self.item_num).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        self.printer = Checker(self.logger)
        
        # Take the pre-trained model
        model_path = f"save_model/{self.args.dataset}/{self.args.model}_{self.args.dataset}.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model.eval()
        
        # Evaluate
        result_5 = self.evaluate(self.model, k=5, is_valid='test')
        result_10 = self.evaluate(self.model, k=10, is_valid='test')
        self.printer.refine_test_result(result_5, result_10)
        self.printer.print_result()



    def evaluate(self, model, k, is_valid='test'):
        """
        Evaluation on validation or test set
        """
        HIT = 0.0 # Overall Hit
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
        
        loader = self.test_loader if is_valid == 'test' else self.valid_loader
        
        for _, (u, seq, item_idx, test_idx) in enumerate(loader):
            u_head = (self.u_head_set[None, ...] == u.numpy()[...,None]).nonzero()[0]         # Index of head users
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)                                  # Index of tail users
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[...,None]).nonzero()[0]  # Index of head items
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)                                  # Index of tail items
            
            predictions = -model.predict(seq.numpy(), item_idx.numpy(), u_tail) # Sequence Encoder
            
            rank = predictions.argsort(1).argsort(1)[:,0].cpu().numpy()
            n_all_user += len(predictions)
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
            

        result = {'Overall': {'NDCG': NDCG / n_all_user, 'HIT': HIT / n_all_user}, 
                'Head_User': {'NDCG': HEAD_USER_NDCG / n_head_user, 'HIT': HEAD_USER_HIT / n_head_user},
                'Tail_User': {'NDCG': TAIL_USER_NDCG / n_tail_user, 'HIT': TAIL_USER_HIT / n_tail_user},
                'Head_Item': {'NDCG': HEAD_ITEM_NDCG / n_head_item, 'HIT': HEAD_ITEM_HIT / n_head_item},
                'Tail_Item': {'NDCG': TAIL_ITEM_NDCG / n_tail_item, 'HIT': TAIL_ITEM_HIT / n_tail_item}
                }

        return result



class SASRec(torch.nn.Module):
    """
    Parameter of SASRec
    """
    def __init__(self, args, item_num):
        super(SASRec, self).__init__()
        self.args = args
        self.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu"

        self.item_emb = torch.nn.Embedding(item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)



    def log2feats(self, log_seqs):
        """
        Sequence Encoder: f_{\theta}(S_u)
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.device)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                         
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats



    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.device))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.device))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)


        return pos_logits, neg_logits



    def predict(self, log_seqs, item_indices, u_tail, u_transfer = None):
        """
        MELT - Prediction
        """
        log_feats = self.log2feats(log_seqs)

        final_feat = log_feats[:, -1, :]
        if u_transfer:
            # Knowledge transfer to tail users
            final_feat[u_tail] = u_transfer(final_feat[u_tail]) + final_feat[u_tail] 

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.device)) 

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits


    def update_tail_item_representation(self, i_tail_loader, item_context, W_I):
        """
        Update all the tail item embeddings
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
            group_context_embed = self.user_representation(np.vstack(collect_context))
            i_tail_average_emb = []
            idx = 0 
            for i in i_t[0]:
                i = i.item()
                if len(item_context[i]) >=1:
                    i_encode_emb = group_context_embed[i_tail_idx[idx]:i_tail_idx[idx+1]]
                    i_tail_average_emb.append(i_encode_emb.mean(0))
                    idx+=1
            group_fully_context_embed = torch.stack(i_tail_average_emb)
            i_tail_estimate_embed = W_I(group_fully_context_embed) 
            
            tail_idx = torch.LongTensor(tail_idx).to(self.device)
            self.item_emb.weight.data[tail_idx] = i_tail_estimate_embed # Direclty update the tail item embedding
    
    
    
    def user_representation(self, log_seqs):
        """
        User representation
        """
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        return final_feat



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs