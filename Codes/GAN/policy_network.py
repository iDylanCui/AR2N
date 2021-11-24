import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Transformer import TransformerModel

class Policy_Network(nn.Module):
    def __init__(self, args, word_embeddings, entity_embeddings, relation_embeddings):
        super(Policy_Network, self).__init__()
        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_emb = args.is_train_emb
        self.use_coverage_attn = args.use_coverage_attn

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.encoder_dropout_rate = args.encoder_dropout_rate

        self.relation_only = args.relation_only
        self.history_dim = args.history_dim
        self.history_layers = args.history_layers
        self.rl_dropout_rate = args.rl_dropout_rate
        self.history_layers = args.history_layers

        self.word_embeddings = Embedding(word_embeddings, self.word_dropout_rate, self.is_train_emb, self.word_padding_idx)

        self.entity_embeddings = Embedding(entity_embeddings, self.emb_dropout_rate, self.is_train_emb)
        self.relation_embeddings = Embedding(relation_embeddings, self.emb_dropout_rate, self.is_train_emb)

        self.Transformer = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate) 

        self.question_linear = nn.Linear(self.word_dim, self.relation_dim)
        self.relation_linear = nn.Linear(self.relation_dim, self.relation_dim)
        self.coverage_linear = nn.Linear(1, self.relation_dim)
        
        self.W_att = nn.Linear(self.relation_dim, 1)

        self.input_dim = self.history_dim + self.word_dim
        
        if self.relation_only:
            self.action_dim = self.relation_dim
        else:
            self.action_dim = self.relation_dim + self.entity_dim
        
        self.lstm_input_dim = self.action_dim 
        
        self.W1 = nn.Linear(self.input_dim, self.action_dim)
        self.W2 = nn.Linear(self.action_dim, self.action_dim)
        self.W1Dropout = nn.Dropout(self.rl_dropout_rate)
        self.W2Dropout = nn.Dropout(self.rl_dropout_rate)

        self.use_entity_embedding_in_vn = args.use_entity_embedding_vn
        if self.use_entity_embedding_in_vn:
            self.W_value = nn.Linear(self.history_dim + self.entity_dim, 1)
        else:
            self.W_value = nn.Linear(self.history_dim, 1) 

        self.path_encoder = nn.LSTM(input_size=self.lstm_input_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_layers,
                                    batch_first=True) 
        
        self.initialize_modules()

    # embedding -> transformer encoder
    def get_question_representation(self, batch_question, batch_sent_len):
        
        batch_question_embedding = self.word_embeddings(batch_question) # [batch, max_tokens, word_embeddings]
        
        mask = self.batch_sentence_mask(batch_sent_len) # [batch, seq_len]
        transformer_output = self.Transformer(batch_question_embedding.permute(1, 0 ,2), mask)
        
        transformer_output = transformer_output.permute(1, 0 ,2) # [batch, seq_size, 300]

        return transformer_output, mask
    
    def get_relation_aware_question_vector_metaqa(self, b_question_vector, b_question_mask, b_r_space, b_coverage_t):
        relation_num, _ = self.relation_embeddings.embedding.weight.shape
        b_size, seq_len = b_question_mask.shape # [b_batch, seq_len]
        
        b_question_vector = b_question_vector.unsqueeze(1).repeat(1, relation_num, 1, 1).view(b_size * relation_num, seq_len, -1) # [b_size * relation_num, seq_len, hidden]
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, relation_num, 1).view(b_size * relation_num, seq_len) # [b_size * relation_num, seq_len]

        b_question_project = self.question_linear(b_question_vector) # [batch * relation_num, seq_size, relation_hidden]
        b_relation_vector = self.relation_embeddings.embedding.weight.unsqueeze(1).unsqueeze(0).repeat(b_size, 1, seq_len, 1).view(b_size * relation_num, seq_len, -1) # [b_size * relation_num, seq_len, hidden]
        b_relation_project = self.relation_linear(b_relation_vector)

        b_att_features = b_question_project + b_relation_project

        if b_coverage_t is not None:
            b_coverage_t = b_coverage_t.unsqueeze(1).repeat(1, relation_num, 1).view(b_size * relation_num, seq_len, 1) # [b_batch * relation_num, seq_len, 1]
            b_coverage_project = self.coverage_linear(b_coverage_t) # [b_batch * relation_num, seq_len, relation_hidden]
            b_att_features += b_coverage_project
        
        # compute attention score and normalize them
        b_att_features_tanh = torch.tanh(b_att_features)
        b_linear_result = self.W_att(b_att_features_tanh).squeeze(-1) # [b_batch * relation_num, seq_len]

        b_linear_result_masked = b_linear_result.masked_fill(b_question_mask, float('-inf'))
        b_matrix_alpha = F.softmax(b_linear_result_masked, 1).unsqueeze(1) # [b_batch * relation_num, 1, seq_len]
        b_relation_aware_question_vector = torch.matmul(b_matrix_alpha, b_question_vector).squeeze(1).view(b_size, relation_num, -1) # [b_batch, relation_num, relation_dim]

        b_matrix_alpha = b_matrix_alpha.squeeze(1).view(b_size, relation_num, -1) # [b_batch, relation_num, seq_len]

        l_relation_aware_question_vector = []
        l_matrix_alpha = []

        for batch_i in range(b_size):
            output_i = b_relation_aware_question_vector[batch_i] # [relation_num, relation_dim]
            matrix_i = b_matrix_alpha[batch_i]
            relation_i = b_r_space[batch_i] # [action_num]
            new_output_i = output_i[relation_i]
            new_matrix_i = matrix_i[relation_i]

            l_relation_aware_question_vector.append(new_output_i)
            l_matrix_alpha.append(new_matrix_i)
        
        b_relation_aware_question_vector = torch.stack(l_relation_aware_question_vector, 0)
        b_matrix_alpha = torch.stack(l_matrix_alpha, 0)

        return b_relation_aware_question_vector, b_matrix_alpha # [batch, action_num, relation_dim]
    
    def get_relation_aware_question_vector(self, b_question_vector, b_question_mask, b_r_space, b_coverage_t):
        b_size, action_num = b_r_space.shape # [b_batch, action_num]
        _, seq_len = b_question_mask.shape # [b_batch, seq_len]
        
        b_question_vector = b_question_vector.unsqueeze(1).repeat(1, action_num, 1, 1).view(b_size * action_num, seq_len, -1) # [b_size * action_num, seq_len, hidden]
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, seq_len) # [b_size * action_num, seq_len]

        b_question_project = self.question_linear(b_question_vector) # [batch * action_num, seq_size, relation_hidden] 
        b_relation_vector = self.relation_embeddings(b_r_space) # [b_batch, action_num, relation_hidden]
        b_relation_vector = b_relation_vector.unsqueeze(2).repeat(1, 1, seq_len, 1).view(b_size * action_num, seq_len, -1) # [b_batch * action_num, seq_len, relation_hidden]
        b_relation_project = self.relation_linear(b_relation_vector)

        b_att_features = b_question_project + b_relation_project

        if b_coverage_t is not None: # coverage loss
            b_coverage_t = b_coverage_t.unsqueeze(1).repeat(1, action_num, 1).view(b_size * action_num, seq_len, 1) # [b_batch *  action_num, seq_len, 1]
            b_coverage_project = self.coverage_linear(b_coverage_t) # [b_batch *  action_num, seq_len, relation_hidden]
            b_att_features += b_coverage_project
        
        # compute attention score and normalize them
        b_att_features_tanh = torch.tanh(b_att_features)
        b_linear_result = self.W_att(b_att_features_tanh).squeeze(-1) # [b_batch * action_num, seq_len]

        b_linear_result_masked = b_linear_result.masked_fill(b_question_mask, float('-inf'))
        b_matrix_alpha = F.softmax(b_linear_result_masked, 1).unsqueeze(1) # [batch_size * action_num, 1, seq_len]
        b_relation_aware_question_vector = torch.matmul(b_matrix_alpha, b_question_vector).squeeze(1).view(b_size, action_num, -1)

        return b_relation_aware_question_vector, b_matrix_alpha.squeeze(1).view(b_size, action_num, -1) # [batch, action_num, relation_dim]

    def get_action_space_in_buckets(self, batch_e_t, d_entity2bucketid, d_action_space_buckets):
        db_action_spaces, db_references = [], []

        entity2bucketid = d_entity2bucketid[batch_e_t.tolist()] 
        key1 = entity2bucketid[:, 0] 
        key2 = entity2bucketid[:, 1] 
        batch_ref = {}

        for i in range(len(batch_e_t)):
            key = int(key1[i])
            if not key in batch_ref:
                batch_ref[key] = []
            batch_ref[key].append(i) 

        # key: the number of actions 
        for key in batch_ref:
            action_space = d_action_space_buckets[key] 
            l_batch_refs = batch_ref[key] 
            g_bucket_ids = key2[l_batch_refs].tolist() 
            r_space_b = action_space[0][0][g_bucket_ids]
            e_space_b = action_space[0][1][g_bucket_ids]
            action_mask_b = action_space[1][g_bucket_ids]

            r_space_b = r_space_b.cuda()
            e_space_b = e_space_b.cuda()
            action_mask_b = action_mask_b.cuda()

            action_space_b = ((r_space_b, e_space_b), action_mask_b)
            db_action_spaces.append(action_space_b)
            db_references.append(l_batch_refs)

        return db_action_spaces, db_references
    
    def policy_linear(self, b_input_vector): # [b_batch, action_num, history_hidden + word_hidden]
        X = self.W1(b_input_vector) # [b_batch, action_num, action_dim] 
        X = F.relu(X)
        X = self.W1Dropout(X)
        X = self.W2(X)
        X2 = self.W2Dropout(X) # [b_batch, action_num, action_dim]
        return X2
    
    def get_action_embedding(self, action):
        r, e = action
        relation_embedding = self.relation_embeddings(r) # [l_batch, action_num, emb_dim]
        
        if self.relation_only:
            action_embedding = relation_embedding
        else:
            entity_embedding = self.entity_embeddings(e) # [l_batch, action_num, emb_dim]
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1) # [l_batch, action_num, emb_dim * 2]
        return action_embedding
    
    def apply_action_masks(self, t, last_r, r_space, action_mask, max_hop, NO_OP_RELATION = 2): 
        if t == 0:
            judge = (r_space == NO_OP_RELATION).long()
            judge = 1 - judge
            action_mask = judge * action_mask
        
        elif t == max_hop - 1:
            judge = (r_space == NO_OP_RELATION).long()
            action_mask = judge * action_mask
        
        else:
            judge = (last_r == NO_OP_RELATION).long() # [batch]
            judge = 1 - judge
            action_mask = judge.unsqueeze(1) * action_mask
            self_loop = torch.zeros(action_mask.size(), dtype= torch.long)
            self_loop = self_loop.cuda()

            self_loop[:, 0] = 1
            self_loop = (1 - judge).unsqueeze(1) * self_loop
            action_mask = action_mask + self_loop
        
        return action_mask
    
    def get_golden_label(self, b_golden_relation_t, b_golden_entity_t, b_r_space, b_e_space):
        b_size, action_num = b_r_space.shape
        relation_equal = (b_r_space == b_golden_relation_t).long()
        entity_equal = (b_e_space == b_golden_entity_t).long()
        b_golden_label_t = relation_equal * entity_equal

        assert torch.sum(b_golden_label_t).item() == b_size 
        
        return b_golden_label_t # [b_batch, action_num]

    def transit(self, t, batch_e_t, batch_question, batch_sent_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, coverage_t, max_hop, dataset, use_mask_trick = False): 
        '''
        t: int
        batch_e_head = batch_e_t = batch_sent_len = [batch]
        batch_question = [batch, seq_len]
        batch_path_hidden = [batch, history_dim]
        '''

        batch_question_vector, batch_question_mask = self.get_question_representation(batch_question, batch_sent_len) # [batch, seq_len, word_dim]

        db_action_spaces, db_references = self.get_action_space_in_buckets(batch_e_t, d_entity2bucketid, d_action_space_buckets)

        db_outcomes = []
        l_attention = []
        references = []
        values_list = []

        for b_action_space, b_reference in zip(db_action_spaces, db_references): 
            b_last_r = last_r[b_reference]
            b_e_t = batch_e_t[b_reference]
            b_path_hidden = batch_path_hidden[b_reference] # [b_batch, history_dim]
            (b_r_space, b_e_space), b_action_mask = b_action_space

            b_size, action_num = b_r_space.shape
            b_question_vector = batch_question_vector[b_reference]
            b_question_mask = batch_question_mask[b_reference]
            if coverage_t is not None:
                b_coverage_t = coverage_t[b_reference]
            else:
                b_coverage_t = None
            
            if self.use_entity_embedding_in_vn:
                b_e_t_embeddings = self.entity_embeddings(b_e_t) # [b_batch, entity_dim]
                value_input = torch.cat([b_e_t_embeddings, b_path_hidden], dim = -1) # [b_batch, history_dim + entity_dim]
                b_value = self.W_value(value_input).view(-1)
            else:
                b_value = self.W_value(b_path_hidden).view(-1)

            b_value = torch.sigmoid(b_value)
            values_list.append(b_value)
            
            if dataset.startswith("MetaQA"):
                b_relation_aware_question_vector, b_attention_matrix = self.get_relation_aware_question_vector_metaqa(b_question_vector, b_question_mask, b_r_space, b_coverage_t) 
            else:
                b_relation_aware_question_vector, b_attention_matrix = self.get_relation_aware_question_vector(b_question_vector, b_question_mask, b_r_space, b_coverage_t)
            
            b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1) # [b_batch, action_num, history_dim]
            b_input_vector = torch.cat([b_path_hidden, b_relation_aware_question_vector], -1) # # [b_batch, action_num, history_dim + word_dim]

            b_output_vector = self.policy_linear(b_input_vector) # [b_batch, action_num, action_dim]
            
            b_action_embedding = self.get_action_embedding((b_r_space, b_e_space)) # [b_batch, action_num, action_dim]

            b_action_embedding = b_action_embedding.view(-1, self.action_dim).unsqueeze(1) # [b_batch * action_num, 1, action_dim]
            b_output_vector = b_output_vector.view(-1, self.action_dim).unsqueeze(-1) # [b_batch * action_num, action_dim, 1]
            b_action_score = torch.matmul(b_action_embedding, b_output_vector).squeeze(-1).view(-1, action_num) # [l_batch, action_num]
            
            if use_mask_trick:
                b_action_mask = self.apply_action_masks(t, b_last_r, b_r_space, b_action_mask, max_hop)
                
            b_action_score_masked = b_action_score.masked_fill((1- b_action_mask).bool(), float('-inf'))
            b_action_dist = F.softmax(b_action_score_masked, 1)

            b_action_space = ((b_r_space, b_e_space), b_action_mask)
            references.extend(b_reference)
            db_outcomes.append((b_action_space, b_action_dist))
            l_attention.append(b_attention_matrix)
        
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        
        return db_outcomes, l_attention, values_list, inv_offset

    def initialize_path(self, init_action):
        init_action_embedding = self.get_action_embedding(init_action)
        init_action_embedding.unsqueeze_(1) # [batch, 1, emb_dim * 2]
        # [num_layers, batch_size, dim]
        init_h = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim]) # (num_layers, batch, hidden_size)
        init_c = torch.zeros([self.history_layers, len(init_action_embedding), self.history_dim])

        init_h = init_h.cuda()
        init_c = init_c.cuda()

        h_n, c_n = self.path_encoder(init_action_embedding, (init_h, init_c))[1]
        return (h_n, c_n) # [num_layers, batch, hidden_dim]

    def transit_pretrain(self, t, batch_e_t, batch_question, batch_sent_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, coverage_t, max_hop, golden_action_t, batch_hops_mask_t, dataset, use_mask_trick = False):
        '''
        t: int
        batch_e_head = batch_e_t = batch_sent_len = [batch]
        batch_question = [batch, seq_len]
        batch_path_hidden = [batch, history_dim]
        '''
        golden_relation_t, golden_entity_t = golden_action_t # [batch, 1]

        batch_question_vector, batch_question_mask = self.get_question_representation(batch_question, batch_sent_len) # [batch, seq_len, word_dim]

        db_action_spaces, db_references = self.get_action_space_in_buckets(batch_e_t, d_entity2bucketid, d_action_space_buckets) 
        db_outcomes = []
        references = []
        l_attention = []
        l_golden_label = []
        batch_nll_loss_t = 0.0

        for b_action_space, b_reference in zip(db_action_spaces, db_references):
            b_last_r = last_r[b_reference]
            b_path_hidden = batch_path_hidden[b_reference] # [b_batch, history_dim]
            b_hops_mask_t = batch_hops_mask_t[b_reference] # [b_batch]

            (b_r_space, b_e_space), b_action_mask = b_action_space
            b_size, action_num = b_r_space.shape
            b_question_vector = batch_question_vector[b_reference]
            b_question_mask = batch_question_mask[b_reference]
            if coverage_t is not None:
                b_coverage_t = coverage_t[b_reference]
            else:
                b_coverage_t = None
            
            if dataset.startswith("MetaQA"):
                b_relation_aware_question_vector, b_attention_matrix = self.get_relation_aware_question_vector_metaqa(b_question_vector, b_question_mask, b_r_space, b_coverage_t) 
            else:
                b_relation_aware_question_vector, b_attention_matrix = self.get_relation_aware_question_vector(b_question_vector, b_question_mask, b_r_space, b_coverage_t)
            
            b_path_hidden = b_path_hidden.unsqueeze(1).repeat(1, action_num, 1) # [b_batch, action_num, history_dim]
            b_input_vector = torch.cat([b_path_hidden, b_relation_aware_question_vector], -1) # # [l_batch, action_num, history_dim + word_dim]
            
            b_output_vector = self.policy_linear(b_input_vector) # [b_batch, action_num, action_dim]

            b_action_embedding = self.get_action_embedding((b_r_space, b_e_space)) # [b_batch, action_num, action_dim]

            b_action_embedding = b_action_embedding.view(-1, self.action_dim).unsqueeze(1) # [b_batch * action_num, 1, action_dim]
            b_output_vector = b_output_vector.view(-1, self.action_dim).unsqueeze(-1) # [b_batch * action_num, action_dim, 1]
            b_action_score = torch.matmul(b_action_embedding, b_output_vector).squeeze(-1).view(-1, action_num) # [l_batch, action_num]
            
            if use_mask_trick:
                b_action_mask = self.apply_action_masks(t, b_last_r, b_r_space, b_action_mask, max_hop)
            
            b_action_score_masked = b_action_score.masked_fill((1- b_action_mask).bool(), float('-inf'))
            b_action_dist = F.softmax(b_action_score_masked, 1)

            b_golden_relation_t = golden_relation_t[b_reference] # [b_batch, 1]
            b_golden_entity_t = golden_entity_t[b_reference] # [b_batch, 1]
            b_golden_label_t = self.get_golden_label(b_golden_relation_t, b_golden_entity_t, b_r_space, b_e_space) # [b_batch, action_num]

            b_golden_id_t = b_last_r.new_zeros(*b_last_r.size()) # golden action [b_batch]
            b_golden_label_t_nonzero = torch.nonzero(b_golden_label_t).cpu().numpy().tolist()
            for position in b_golden_label_t_nonzero:
                x, y = position
                b_golden_id_t[x] = y
            
            b_golden_id_t = b_golden_id_t.view(-1, 1) # [b_batch, 1]

            b_losses_mask = self.masked_nll_loss(b_action_dist, b_golden_id_t, b_hops_mask_t)
            batch_nll_loss_t += torch.sum(b_losses_mask)

            b_action_space = ((b_r_space, b_e_space), b_action_mask)
            references.extend(b_reference)
            db_outcomes.append((b_action_space, b_action_dist))
            l_attention.append(b_attention_matrix)
            l_golden_label.append(b_golden_id_t)
            
        inv_offset = [i for i, _ in sorted(enumerate(references), key=lambda x: x[1])]
        return db_outcomes, l_attention, l_golden_label, batch_nll_loss_t, inv_offset

    def masked_nll_loss(self, b_action_dist, b_golden_id_t, b_hops_mask_t):
        '''
        b_action_dist: [b_batch, action_num]
        b_golden_id_t: [b_batch, 1]
        b_hops_mask_t: [b_batch]
        '''
        EPS = 1e-8
        b_log_action_dist = torch.log(b_action_dist + EPS)
        b_losses = -torch.gather(b_log_action_dist, dim=1, index=b_golden_id_t).squeeze(1)  # [b_batch]
        b_losses_mask = b_losses * b_hops_mask_t
        return b_losses_mask

    def update_path(self, action, path_list, offset=None):
        
        def offset_path_history(p, offset):
            for i, x in enumerate(p):
                if type(x) is tuple:
                    new_tuple = tuple([_x[:, offset, :] for _x in x])
                    p[i] = new_tuple
                else:
                    p[i] = x[offset, :]

        # update action history
        action_embedding = self.get_action_embedding(action)
        action_embedding.unsqueeze_(1) # [batch, 1, emb_dim * 2]
        
        if offset is not None:
            offset_path_history(path_list, offset)

        h_n, c_n = self.path_encoder(action_embedding, path_list[-1])[1]
        return path_list, (h_n, c_n)
    
    def update_coverage(self, attention_matrix, l_coverage, offset=None):
        
        def offset_coverage_history(p, offset):
            for i, x in enumerate(p):
                p[i] = x[offset, :]
        
        if offset is not None:
            offset_coverage_history(l_coverage, offset)

        new_coverage = attention_matrix + l_coverage[-1]
        return l_coverage, new_coverage

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.question_linear.weight)
        nn.init.constant_(self.question_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.relation_linear.weight)
        nn.init.constant_(self.relation_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.coverage_linear.weight)
        nn.init.constant_(self.coverage_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.constant_(self.W_att.bias, 0.0)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.constant_(self.W1.bias, 0.0)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.constant_(self.W2.bias, 0.0)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.constant_(self.W_value.bias, 0.0)

        for name, param in self.path_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def batch_sentence_mask(self, batch_sent_len):
        batch_size = len(batch_sent_len)
        max_sent_len = batch_sent_len[0]
        mask = torch.zeros(batch_size, max_sent_len, dtype=torch.long)

        for i in range(batch_size):
            sent_len = batch_sent_len[i]
            mask[i][sent_len:] = 1
        
        mask = (mask == 1)
        mask = mask.cuda()
        return mask
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir, map_location={'cuda:1':'cuda:0'}))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)