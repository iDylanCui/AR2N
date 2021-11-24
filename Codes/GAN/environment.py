import numpy as np
import torch

class Environment(object):
    def __init__(self, args, is_Pretrain_Gen, isTrain, dummy_start_r = 1):
        self.is_Pretrain_Gen = is_Pretrain_Gen
        self.isTrain = isTrain
        self.dummy_start_r = dummy_start_r
        self.use_coverage_attn = args.use_coverage_attn
        

    def reset(self, bath_data, generator):
        if self.is_Pretrain_Gen and self.isTrain:
            self.batch_question, self.batch_question_len, self.batch_head, self.batch_answers, self.batch_relation_path, self.batch_relation_lengths, self.batch_intermediate_entities = bath_data # self.batch_relation_path: [batch, max_hop]
        else:
            self.batch_question, self.batch_question_len, self.batch_head, self.batch_answers, self.batch_relation_path, self.batch_relation_lengths = bath_data

        r_s = torch.zeros_like(self.batch_head) + self.dummy_start_r 
        self.path_trace = [(r_s, self.batch_head)]
        self.path_hidden = [generator.initialize_path(self.path_trace[0])]
        
        _, self.max_hop = self.batch_relation_path.shape

        if self.use_coverage_attn:
            self.l_coverage = [torch.zeros_like(self.batch_question, dtype=torch.float)]
            self.l_attention = [torch.zeros_like(self.batch_question, dtype=torch.float)]

    def observe(self):
        if self.use_coverage_attn:
            return self.path_trace, self.path_hidden, self.l_coverage
        else:
            return self.path_trace, self.path_hidden

    def step(self, *update_items):
        if len(update_items) == 3:
            new_action, path_hidden, new_hidden = update_items
            self.path_trace.append(new_action)
            self.path_hidden = path_hidden
            self.path_hidden.append(new_hidden)
        elif len(update_items) == 6:
            new_action, path_hidden, new_hidden, l_coverage_new, new_coverage, attention_matrix = update_items
            self.path_trace.append(new_action)
            self.path_hidden = path_hidden
            self.path_hidden.append(new_hidden)
            
            self.l_coverage = l_coverage_new
            self.l_coverage.append(new_coverage)
            self.l_attention.append(attention_matrix)
    
    def get_pred_entities(self):
        return self.path_trace[-1][1]
    
    def return_batch_data(self):
        if self.is_Pretrain_Gen and self.isTrain:
            return self.batch_question, self.batch_question_len, self.batch_head, self.batch_relation_path, self.batch_relation_lengths, self.batch_intermediate_entities
        else:
            return self.batch_question, self.batch_question_len, self.batch_head, self.batch_answers, self.batch_relation_path, self.batch_relation_lengths
    
    def return_coverage_loss(self, batch_hops_mask): 
        if self.use_coverage_attn:
            coverage_loss = 0.0
            for i in range(1, self.max_hop+1):
                batch_hops_mask_t = batch_hops_mask[:, i-1]
                coverage_t_1 = self.l_coverage[i-1]
                attention_t = self.l_attention[i]
                coverage_loss_t = torch.min(coverage_t_1, attention_t)
                coverage_loss_t_sum = torch.sum(coverage_loss_t, dim = 1)
                coverage_loss_t_sum_mask = batch_hops_mask_t * coverage_loss_t_sum

                coverage_loss += torch.sum(coverage_loss_t_sum_mask)
            
            return coverage_loss
    
    def get_pred_entities(self):
        return self.path_trace[-1][1]

    def set_path_trace(self, path_trace):
        self.path_trace = path_trace
    
    def set_path_hidden(self, path_hidden):
        self.path_hidden = path_hidden
    
    def set_l_coverage(self, l_coverage):
        self.l_coverage = l_coverage
    
    def set_l_attention(self, l_attention):
        self.l_attention = l_attention

    def set_max_hop(self, max_hop):
        self.max_hop = max_hop
    
    def get_l_attention(self):
        return self.l_attention

    def get_path_trace(self):
        return self.path_trace