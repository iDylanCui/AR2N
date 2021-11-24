import torch
from utils import pad_and_cat, pad_and_cat2d


def top_k_action(log_action_dist, action_space, batch_attention_padding, batch_size, beam_size):
    full_size = len(log_action_dist)
    last_k = int(full_size / batch_size)

    (r_space, e_space), _ = action_space
    action_space_size = r_space.size()[1]
    _, _, seq_len =  batch_attention_padding.shape # [batch*k, action_num, seq_len]
    batch_attention_padding = batch_attention_padding.view(batch_size, -1, seq_len) # [batch, k*action_num, seq_len]
    new_attention_list = []

    log_action_dist = log_action_dist.view(batch_size, -1) # [batch_size, k*action_space_size]
    beam_action_space_size = log_action_dist.size()[1]
    k = min(beam_size, beam_action_space_size) 

    log_action_prob, action_ind = torch.topk(log_action_dist, k) # [batch_size, k]

    next_r = torch.gather(r_space.view(batch_size, -1), 1, action_ind).view(-1) # [batch_size* k]
    next_e = torch.gather(e_space.view(batch_size, -1), 1, action_ind).view(-1) # [batch_size* k]
    log_action_prob = log_action_prob.view(-1) # [batch_size*k]
    
    for b_index in range(batch_size):
        action_indices = action_ind[b_index]
        att_vector = batch_attention_padding[b_index][action_indices] # [k, seq_len]
        new_attention_list.append(att_vector) 
    
    attention_matrix = torch.stack(new_attention_list, dim=0).view(-1, seq_len) # [batch_size*k, seq_len]
    
    # *** compute parent offset
    action_beam_offset = action_ind // action_space_size # [batch_size, k]
    action_batch_offset = (torch.arange(batch_size).cuda() * last_k).unsqueeze(1) # [batch_size, 1]
    # [batch_size, k] => [batch_size*k]
    action_offset = (action_batch_offset + action_beam_offset).view(-1) 

    return (next_r, next_e), log_action_prob, attention_matrix, action_offset # [batch_size * k]

def top_k_answer_unique(log_action_dist, action_space, batch_attention_padding, batch_size, beam_size):
    full_size = len(log_action_dist)
    last_k = int(full_size / batch_size)
    (r_space, e_space), _ = action_space
    action_space_size = r_space.size()[1]

    r_space = r_space.view(batch_size, -1) # [batch, k * action_num]
    e_space = e_space.view(batch_size, -1)
    log_action_dist = log_action_dist.view(batch_size, -1)
    beam_action_space_size = log_action_dist.size()[1]
    _, _, seq_len =  batch_attention_padding.shape # [batch*k, action_num, seq_len]
    batch_attention_padding = batch_attention_padding.view(batch_size, -1, seq_len) # [batch, k*action_num, seq_len]
    new_attention_list = []
    
    k = min(beam_size, beam_action_space_size)
    next_r_list, next_e_list = [], []
    log_action_prob_list = []
    action_offset_list = []

    for i in range(batch_size):
        log_action_dist_b = log_action_dist[i]
        r_space_b = r_space[i]
        e_space_b = e_space[i]
        attention_padding_b = batch_attention_padding[i] 
        unique_e_space_b = torch.unique(e_space_b.data.cpu()).cuda() 
        unique_log_action_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, log_action_dist_b)
        k_prime = min(len(unique_e_space_b), k)
        top_unique_log_action_dist, top_unique_idx2 = torch.topk(unique_log_action_dist, k_prime)
        top_unique_idx = unique_idx[top_unique_idx2]
        top_unique_beam_offset = top_unique_idx // action_space_size
        top_r = r_space_b[top_unique_idx]
        top_e = e_space_b[top_unique_idx]

        att_vector = attention_padding_b[top_unique_idx] # [k, seq_len]
        new_attention_list.append(att_vector.unsqueeze(0)) 

        next_r_list.append(top_r.unsqueeze(0))
        next_e_list.append(top_e.unsqueeze(0))
        log_action_prob_list.append(top_unique_log_action_dist.unsqueeze(0))
        top_unique_batch_offset = i * last_k
        top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
        action_offset_list.append(top_unique_action_offset.unsqueeze(0))
    next_r = pad_and_cat(next_r_list, padding_value=0).view(-1)
    next_e = pad_and_cat(next_e_list, padding_value=0).view(-1)
    attention_matrix = pad_and_cat2d(new_attention_list, padding_value=0).view(-1, seq_len)
    log_action_prob = pad_and_cat(log_action_prob_list, padding_value = -float("inf"))
    action_offset = pad_and_cat(action_offset_list, padding_value=-1)
    
    return (next_r, next_e), log_action_prob.view(-1), attention_matrix, action_offset.view(-1)

def unique_max(unique_x, x, values, marker_2D=None):
    unique_interval = 100
    HUGE_INT = 1e31

    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float() 
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT # [unique_interval, * len(x)]
        unique_values_b, unique_idx_b = values_2D.max(dim=1) 
        unique_values.append(unique_values_b)
        unique_indices.append(unique_idx_b)
    unique_values = torch.cat(unique_values).cuda()
    unique_idx = torch.cat(unique_indices).cuda()
    return unique_values, unique_idx # [len(unique_x)]

