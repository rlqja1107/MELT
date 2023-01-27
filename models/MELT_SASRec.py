import os
import numpy as np
import torch
import torch.utils.data as data_utils
from embedder import embedder
from models.SASRec import SASRec, Trainer
from src.utils import setupt_logger, set_random_seeds, Checker
from src.sampler import NegativeSampler
from src.data import ValidData, TestData, TrainData

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
        i_L_max = self.item_threshold + 50
        self.model = MELT(self.args, self.train_data, self.device, self.item_num, u_L_max, i_L_max).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)
        
        # Build the train, valid, test loader
        train_dataset = TrainData(self.train_data, self.user_num, self.item_num, batch_size=self.args.batch_size, maxlen=self.args.maxlen)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        valid_dataset = ValidData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.valid_loader = data_utils.DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        
        # For selecting the best model
        self.validcheck = Checker(self.logger)
        
        adam_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        
        i_tail_set = data_utils.TensorDataset(torch.LongTensor(list(self.i_tail_set)))
        
        # For updating the tail item embedding
        i_tail_loader = data_utils.DataLoader(i_tail_set, self.args.batch_size * 2, shuffle=False, drop_last=False) # No matter how batch size is
        
        # DataLoader for bilateral branches
        i_batch_size = len(self.i_head_set) // (len(train_loader)-1) + 1
        i_h_loader = data_utils.DataLoader(self.i_head_set, i_batch_size, shuffle=True, drop_last=False)
        u_batch_size = len(self.u_head_set) // (len(train_loader)-1) + 1
        u_h_loader = data_utils.DataLoader(self.u_head_set, u_batch_size, shuffle=True, drop_last=False)
        
        for epoch in range(self.args.e_max):
            training_loss = 0.0
            self.model.eval()
            
            with torch.no_grad():
                # Knowledge transfer from item branch to user branch
                self.model.sasrec.update_tail_item_representation(i_tail_loader, self.item_context, self.model.item_branch.W_I) 
                #i_tail_loader, item_context, W_I
                
            self.model.train()
            for _, ((u,seq,pos,neg),(u_idx), (i_idx)) in enumerate(zip(train_loader, u_h_loader, i_h_loader)):
                adam_optimizer.zero_grad()
                u = np.array(u); seq = np.array(seq); pos = np.array(pos); neg = np.array(neg)
                u_idx = u_idx.numpy()
                i_idx = i_idx.numpy()

                loss = self.model(u, seq, pos, neg, u_idx, i_idx, self.user_context, self.item_context, self.n_item_context, \
                                  self.user_threshold, self.item_threshold, epoch)
                loss.backward()
                adam_optimizer.step()
                training_loss += loss.item()

            if epoch % 2 == 0:
                with torch.no_grad():
                    self.model.eval()
                    result_valid = self.evaluate(self.model, k=10, is_valid='valid')
                    best_valid = self.validcheck(result_valid, epoch, self.model, f'{self.args.model}_{self.args.dataset}.pth')
        
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Evaluating: Dataset({self.args.dataset}), Loss: ({training_loss:.2f}), GPU: {self.args.gpu}')
                
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


    def test(self):
        set_random_seeds(self.args.seed)
        user_max_thres = self.args.maxlen
        item_max_thres = self.item_threshold + self.args.maxlen
        self.model = MELT(self.args, self.train_data, self.device, self.item_num, user_max_thres, item_max_thres).to(self.device)
        self.inference_negative_sampler = NegativeSampler(self.args, self.dataset)

        test_dataset = TestData(self.args, self.item_context, self.train_data, self.test_data, self.valid_data, self.inference_negative_sampler)
        self.test_loader = data_utils.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=False)
        self.printer = Checker(self.logger)

        os.makedirs(f"save_model/{self.args.dataset}/{self.args.model}", exist_ok=True)

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
        
        loader = self.test_loader if is_valid == 'test' else self.valid_loader
        
        for _, (u, seq, item_idx, test_idx) in enumerate(loader):
            u_head = (self.u_head_set[None, ...] == u.numpy()[...,None]).nonzero()[0]         # Index of head users
            u_tail = np.setdiff1d(np.arange(len(u)), u_head)                                  # Index of tail users
            i_head = (self.i_head_set[None, ...] == test_idx.numpy()[...,None]).nonzero()[0]  # Index of head items
            i_tail = np.setdiff1d(np.arange(len(u)), i_head)                                  # Index of tail items
            
            predictions = -model.sasrec.predict(seq.numpy(), item_idx.numpy(), u_tail, model.user_branch.W_U) # Sequence Encoder
            
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


    
class MELT(torch.nn.Module):
    def __init__(self, args, train_data, device, item_num, u_L_max, i_L_max):
        super(MELT, self).__init__()
        self.args = args
        self.device = device
        self.user_branch = USERBRANCH(self.args, device, u_L_max)
        self.item_branch = ITEMBRANCH(self.args, self.device, i_L_max)
        self.sasrec = SASRec(args, item_num)
        
        self.load_pretrained_model()
        self.train_data = train_data
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        

    def load_pretrained_model(self):
        try:
            model_path = f"save_model/{self.args.dataset}/SASRec_{self.args.dataset}.pth"
            self.sasrec.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        except:
            sasrec = Trainer(self.args)
            sasrec.train()
            model_path = f"save_model/{self.args.dataset}/SASRec_{self.args.dataset}.pth"
            self.sasrec.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))


    def forward(self, u, seq, pos, neg, u_h_idx, i_h_idx, user_context, item_context, n_item_context, user_thres, item_thres, epoch):
        # User branch loss (L_u)
        user_loss = self.user_branch(u_h_idx, self.sasrec, user_context, user_thres, epoch)
        
        # Item branch loss (L_i)
        item_loss = self.item_branch(i_h_idx, item_context, self.sasrec, item_thres, self.user_branch.W_U, epoch, n_item_context)
        
        # Next item prediction loss (L_{rec})
        pos_logits, neg_logits = self.sasrec(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).to(self.device), torch.zeros(neg_logits.shape).to(self.device)
        indices = np.where(pos != 0)
        prediction_loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
        prediction_loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
        
        loss = user_loss * self.args.lamb_u + item_loss * self.args.lamb_i + prediction_loss
        return loss



class USERBRANCH(torch.nn.Module):
    def __init__(self, args, device, u_L_max):
        """
        User branch: Enhance the tail user representation
        """
        super(USERBRANCH, self).__init__()
        self.args = args
        self.W_U = torch.nn.Linear(self.args.hidden_units, self.args.hidden_units)
        torch.nn.init.xavier_normal_(self.W_U.weight.data)
        self.criterion = torch.nn.MSELoss()
        self.device = device
        self.u_L_max = u_L_max
        self.pi = np.pi
        
        
    def forward(self, u_head_idx, sasrec, user_context, user_thres, epoch):
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
        full_seq_repre = sasrec.user_representation(full_seq)
        few_seq = np.zeros([len(u_head_idx), self.args.maxlen], dtype=np.int32)
  
        R = np.random.randint(1, user_thres, len(full_seq))
        for i, l in enumerate(R):
            few_seq[i, -l:] = full_seq[i, -l:]
            
        # Representations of recent interactions 
        few_seq_repre = sasrec.user_representation(few_seq)
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
        
        
    def forward(self, i_head_idx, item_context, sasrec, item_thres, W_U, epoch=None, n_item_context=None):
        target_embed = []
        subseq_set = []
        subseq_set_idx = [0]
        idx = 0
        w_i_list = []
        
        for i, h_i in enumerate(i_head_idx):
            item_context_list = np.vstack(item_context[h_i])
            n_context = min(self.i_L_max, n_item_context[h_i])
            
            #  Calculate the loss coefficient
            w_i = (self.pi/2)*(epoch/self.args.e_max)+(self.pi/100)*(n_context-(item_thres+1))
            w_i = np.abs(np.sin(w_i))
            w_i_list.append(w_i)
            len_context = len(item_context[h_i])
            
            # Set upper bound of item freq.
            thres = min(len_context, item_thres) 
            n_few_inter = np.random.randint(1, thres+1)
            
            # Randomly sample the contexts
            K = np.random.choice(range(len(item_context_list)), int(n_few_inter), replace=False)
            idx += len(K)
            
            subseq_set.append(item_context_list[K])
            target_embed.append(h_i)
            subseq_set_idx.append(idx)
        
        # Encode the subsequence set
        subseq_repre_set = sasrec.user_representation(np.vstack(subseq_set))
        
        # Knowledge transfer from user to item branch
        subseq_repre_set = subseq_repre_set + W_U(subseq_repre_set) 
        
        # Contextualized representations
        subseq_set = []
        for i, h_i in enumerate(i_head_idx):
            mean_context = subseq_repre_set[subseq_set_idx[i]:subseq_set_idx[i+1]]
            subseq_set.append(mean_context.mean(0))
        
        few_subseq_embed = torch.stack(subseq_set)
        target_embed = torch.LongTensor(target_embed).to(self.device)
        w_i_list = torch.FloatTensor(w_i_list).view(-1,1).to(self.device)
        
        # Curriculum Learning by item
        loss = (w_i_list*((self.W_I(few_subseq_embed) - sasrec.item_emb(target_embed))) ** 2).mean()
        return loss

