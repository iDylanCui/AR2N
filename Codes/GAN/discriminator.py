import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding import Embedding
from Transformer import TransformerModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Discriminator(nn.Module):
    def __init__(self, args, word_embeddings, relation_embeddings):
        super(Discriminator, self).__init__()
        self.word_dim = args.word_dim
        self.word_padding_idx = args.word_padding_idx
        self.word_dropout_rate = args.word_dropout_rate
        self.is_train_emb = args.is_train_emb
        self.use_relation_aware_dis = args.use_relation_aware_dis

        self.relation_dim = args.relation_dim
        self.emb_dropout_rate = args.emb_dropout_rate

        self.encoder_dropout_rate = args.encoder_dropout_rate
        self.head_num = args.head_num
        self.hidden_dim = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.encoder_dropout_rate = args.encoder_dropout_rate

        self.word_embeddings = Embedding(word_embeddings, self.word_dropout_rate, self.is_train_emb, self.word_padding_idx)

        self.relation_embeddings = Embedding(relation_embeddings, self.emb_dropout_rate, self.is_train_emb)

        self.Transformer = TransformerModel(self.word_dim, self.head_num, self.hidden_dim, self.encoder_layers, self.encoder_dropout_rate) 
        
        self.question_linear = nn.Linear(self.word_dim, self.relation_dim)
        self.relation_linear = nn.Linear(self.relation_dim, self.relation_dim)
        self.W_att = nn.Linear(self.relation_dim, 1)

        if self.use_relation_aware_dis:
            self.lstm_input_dim = self.word_dim + self.relation_dim
        else:
            self.lstm_input_dim = self.relation_dim

        self.history_dim = args.history_dim
        self.history_layers = args.history_layers

        self.path_encoder = nn.LSTM(input_size=self.lstm_input_dim,
                                    hidden_size=self.history_dim,
                                    num_layers=self.history_layers,
                                    batch_first=True)
        
        self.question_mp_linear = nn.Linear(self.word_dim, self.lstm_input_dim)
        self.score_linear = nn.Linear(self.history_dim, 1)

        self.initialize_modules()
    
    def get_relation_aware_question_vector(self, b_question_vector, b_question_mask, batch_relation_path_vector):
        batch_size, max_hop, _ = batch_relation_path_vector.shape
        _, seq_len = b_question_mask.shape # [batch_size, seq_len]
        
        b_question_vector = b_question_vector.unsqueeze(1).repeat(1, max_hop, 1, 1).view(batch_size * max_hop, seq_len, -1) #  [batch_size * max_hop, seq_len, word_dim]
        b_question_mask = b_question_mask.unsqueeze(1).repeat(1, max_hop, 1).view(batch_size * max_hop, seq_len) # [b_size * max_hop, seq_len]

        b_question_project = self.question_linear(b_question_vector) # [batch_size * max_hop, seq_size, relation_hidden]

        batch_relation_path_vector = batch_relation_path_vector.unsqueeze(2).repeat(1, 1, seq_len, 1).view(batch_size * max_hop, seq_len, -1) # [batch_size * max_hop, seq_len, relation_hidden]
        b_relation_project = self.relation_linear(batch_relation_path_vector)

        b_att_features = b_question_project + b_relation_project # [batch_size * max_hop, seq_len, relation_hidden]
        
        # compute attention score and normalize them
        b_att_features_tanh = torch.tanh(b_att_features)
        b_linear_result = self.W_att(b_att_features_tanh).squeeze(-1) # [batch_size * max_hop, seq_len]

        b_linear_result_masked = b_linear_result.masked_fill(b_question_mask, float('-inf'))
        b_matrix_alpha = F.softmax(b_linear_result_masked, 1).unsqueeze(1) # [batch_size * max_hop, 1, seq_len]
        b_relation_aware_question_vector = torch.matmul(b_matrix_alpha, b_question_vector).squeeze(1).view(batch_size, max_hop, -1) # [batch, max_hop, word_dim]

        return b_relation_aware_question_vector 
    
    def question_max_pooling(self, transformer_out, question_mask):
        _, _, output_dim = transformer_out.shape
        question_mask = question_mask.unsqueeze(-1).repeat(1,1, output_dim) # [batch, q_len, output_dim]
        transformer_out_masked = transformer_out.masked_fill(question_mask, float('-inf'))
        question_transformer_masked = transformer_out_masked.transpose(1, 2) # [batch, hidden, q_len]
        question_mp = F.max_pool1d(question_transformer_masked, question_transformer_masked.size(2)).squeeze(2)

        return question_mp # [batch, output_dim]
    
    def sort_length_within_batch(self, input, length):
        sorted, idx_sort = torch.sort(length, dim=0, descending=True)
        _, ori_order_idx = torch.sort(idx_sort, dim=0)

        new_input = input[idx_sort]
        new_length = length[idx_sort]

        return new_input, new_length, ori_order_idx
    
    def lstm_calculation(self, input, length):
        batch_size = length.shape[0]

        init_h = torch.zeros([self.history_layers, batch_size, self.history_dim]).cuda() # (num_layers, batch, hidden_size)
        init_c = torch.zeros([self.history_layers, batch_size, self.history_dim]).cuda()
        length = length.cpu()
        input_packed = pack_padded_sequence(input, length, batch_first=True)
        output, _ = self.path_encoder(input_packed, (init_h, init_c))
        output, _ = pad_packed_sequence(output, batch_first=True) # output: [batch, max_hop, hidden_state]
        
        l_hidden_states = []
        for i in range(batch_size):
            last_index = length[i] - 1 
            l_hidden_states.append(output[i][last_index].unsqueeze(0))
        
        batch_hidden_states = torch.cat(l_hidden_states, dim=0) # [batch, hidden]
        return batch_hidden_states

    def reorder(self, rnn_output, ori_order_idx):
        reorder_output = rnn_output[ori_order_idx]
        return reorder_output
    
    def forward(self, batch_question, batch_question_len, batch_relation_path, batch_relation_lengths):
        '''
        batch_question: [batch, seq_len]
        batch_question_len: [batch]
        batch_relation_path: [batch, max_hop]
        batch_relation_lengths: [batch]
        '''
        batch_question_vector, batch_question_mask = self.get_question_representation(batch_question, batch_question_len) # [batch, seq_len, word_dim]

        batch_question_mp = self.question_max_pooling(batch_question_vector, batch_question_mask) # Max-pooling, [batch, 300]

        batch_relation_path_vector = self.relation_embeddings(batch_relation_path) # [batch, max_hop, relation_dim]
        
        if self.use_relation_aware_dis:
            batch_relation_aware_question_vector = self.get_relation_aware_question_vector(batch_question_vector, batch_question_mask, batch_relation_path_vector) # [batch, max_hop, word_dim]

            b_input_vector = torch.cat([batch_relation_path_vector, batch_relation_aware_question_vector], -1) # [batch, max_hop, word_dim + relation_dim]
        else:
            b_input_vector = batch_relation_path_vector # [batch, max_hop, relation_dim]
        
        batch_question_mp_linear = self.question_mp_linear(batch_question_mp).unsqueeze(1)
        batch_input_vector = torch.cat([batch_question_mp_linear, b_input_vector], dim = 1)
        batch_new_path_len = batch_relation_lengths + 1

        batch_input_vector_sorted, batch_new_path_len_sorted, ori_order_idx = self.sort_length_within_batch(batch_input_vector, batch_new_path_len)

        batch_hidden_states_sorted = self.lstm_calculation(batch_input_vector_sorted, batch_new_path_len_sorted) # [batch, hidden]

        batch_hidden_states = self.reorder(batch_hidden_states_sorted, ori_order_idx)

        batch_scores = self.score_linear(batch_hidden_states).view(-1)
        batch_scores_sigmoid = torch.sigmoid(batch_scores)
        
        return batch_scores_sigmoid
    
    def calculate_loss(self, criterion, batch_scores_sigmoid, label):
        batch_labels = batch_scores_sigmoid.new_zeros(*batch_scores_sigmoid.size())
        batch_labels += label
        return criterion(batch_scores_sigmoid, batch_labels)

    
    # embedding -> transformer encoder
    def get_question_representation(self, batch_question, batch_sent_len):
        batch_question_embedding = self.word_embeddings(batch_question) # [batch, max_tokens, word_embeddings]
        
        mask = self.batch_sentence_mask(batch_sent_len) # [batch, seq_len]
        transformer_output = self.Transformer(batch_question_embedding.permute(1, 0 ,2), mask)
        
        transformer_output = transformer_output.permute(1, 0 ,2) 

        return transformer_output, mask

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

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.question_linear.weight)
        nn.init.constant_(self.question_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.relation_linear.weight)
        nn.init.constant_(self.relation_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.W_att.weight)
        nn.init.constant_(self.W_att.bias, 0.0)
        nn.init.xavier_uniform_(self.question_mp_linear.weight)
        nn.init.constant_(self.question_mp_linear.bias, 0.0)
        nn.init.xavier_uniform_(self.score_linear.weight)
        nn.init.constant_(self.score_linear.bias, 0.0)

        for name, param in self.path_encoder.named_parameters(): 
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir, map_location={'cuda:1':'cuda:0'}))

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)