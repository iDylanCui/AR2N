from utils import pad_and_cat_action_space, pad_and_cat, pad_and_cat2d, safe_log, adjust_search_trace, rearrange_vector_list
from beam_search import top_k_action, top_k_answer_unique
import torch
import torch.nn.functional as F

def sample(action_space, action_dist):
    sample_outcome = {}
    (r_space, e_space), action_mask = action_space
    idx = torch.multinomial(action_dist, 1, replacement=True)
    next_r = torch.gather(r_space, 1, idx).view(-1) # [batch_b]
    next_e = torch.gather(e_space, 1, idx).view(-1) # [batch_b]
    action_prob = torch.gather(action_dist, 1, idx).view(-1) # [batch_b]
    sample_outcome['action_sample'] = (next_r, next_e)
    sample_outcome['action_prob'] = action_prob
    return sample_outcome, idx

def sample_action(db_outcomes, l_attention, inv_offset=None, l_golden_label = None):
    next_r_list = []
    next_e_list = []
    action_prob_list = []
    next_attention_list = []
    if l_golden_label is None:
        for (action_space, action_dist), b_attention in zip(db_outcomes, l_attention):
            b_batch, action_num, seq_len = b_attention.shape
            sample_outcome, b_sample_idx = sample(action_space, action_dist)

            for k in range(b_batch):
                action_index = b_sample_idx[k][0]
                att_vector = b_attention[k][action_index]
                next_attention_list.append(att_vector)

            next_r_list.append(sample_outcome['action_sample'][0])
            next_e_list.append(sample_outcome['action_sample'][1])
            action_prob_list.append(sample_outcome['action_prob'])
    
    else:
        for (b_action_space, b_action_dist), b_attention, b_golden_label in zip(db_outcomes, l_attention, l_golden_label): # b_attention: [b_batch, action_num, seq_len], b_golden_label: [b_batch, 1]
            (b_r_space, b_e_space), b_action_mask = b_action_space
            b_batch, action_num, seq_len = b_attention.shape
            b_next_r = torch.gather(b_r_space, 1, b_golden_label).view(-1) # [b_batch]
            b_next_e = torch.gather(b_e_space, 1, b_golden_label).view(-1) 
            b_dist = torch.gather(b_action_dist, 1, b_golden_label).view(-1)
    
            for k in range(b_batch):
                action_index = b_golden_label[k][0]
                att_vector = b_attention[k][action_index]
                next_attention_list.append(att_vector)
            
            next_r_list.append(b_next_r)
            next_e_list.append(b_next_e)
            action_prob_list.append(b_dist)

    next_r = torch.cat(next_r_list, dim=0)[inv_offset] 
    next_e = torch.cat(next_e_list, dim=0)[inv_offset]
    action_sample = (next_r, next_e)
    action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
    attention_matrix = torch.stack(next_attention_list, dim=0)[inv_offset]

    return action_sample, action_prob, attention_matrix

def rollout_normal(args, env, generator, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, end, use_mask_trick = False):
    
    l_log_action_probs = []
    l_action_value = []

    for t in range(0, end):
        if args.use_coverage_attn:
            path_trace, path_hidden, l_coverage = env.observe()
            coverage_t = l_coverage[-1] # [batch, seq_len]
        else:
            path_trace, path_hidden = env.observe()
            coverage_t = None

        last_r, e_t = path_trace[-1]

        batch_path_hidden = path_hidden[-1][0][-1, :, :] # [batch, hidden]
        
        db_outcomes, l_attention, values_list, inv_offset = generator.transit(t, e_t, batch_question, batch_question_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, coverage_t, env.max_hop, args.dataset, use_mask_trick = use_mask_trick) 

        values = torch.cat(values_list, dim=0)[inv_offset]

        action_sample, action_prob, attention_matrix = sample_action(db_outcomes, l_attention, inv_offset, l_golden_label = None)

        l_log_action_probs.append(safe_log(action_prob)) 
        l_action_value.append(values)

        path_list, (h_t, c_t) = generator.update_path(action_sample, path_hidden)
        if args.use_coverage_attn:
            l_coverage, new_coverage = generator.update_coverage(attention_matrix, l_coverage)
            env.step(action_sample, path_list, (h_t, c_t), l_coverage, new_coverage, attention_matrix)
        else:
            env.step(action_sample, path_list, (h_t, c_t))
    
    return l_log_action_probs, l_action_value


def rollout_beam(args, generator, env, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, beam_size, use_mask_trick = False):
    
    log_action_prob = batch_head.new_zeros(*batch_head.size())
    l_search_trace = []
    l_log_action_probs = []

    for t in range(env.max_hop):
        if args.use_coverage_attn:
            path_trace, path_hidden, l_coverage = env.observe()
            coverage_t = l_coverage[-1] # [batch, seq_len]
        else:
            path_trace, path_hidden = env.observe()
            coverage_t = None

        last_r, e_t = path_trace[-1]
        batch_path_hidden = path_hidden[-1][0][-1, :, :] 
        k = int(e_t.size()[0] / batch_size)

        beam_question = batch_question.unsqueeze(1).repeat(1, k, 1).view(batch_size * k, -1) # [batch*k, seq_len]
        beam_question_len = batch_question_len.unsqueeze(1).repeat(1, k).view(batch_size * k) # [batch*k]

        db_outcomes, l_attention, _, inv_offset = generator.transit(t, e_t, beam_question, beam_question_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, coverage_t, env.max_hop, args.dataset, use_mask_trick = use_mask_trick)

        db_action_spaces = [action_space for action_space, _ in db_outcomes]
        db_action_dist = [action_dist for _, action_dist in db_outcomes]

        action_space = pad_and_cat_action_space(db_action_spaces, inv_offset) # [batch*k, padding_action_space]
        action_dist = pad_and_cat(db_action_dist, padding_value=0)[inv_offset]

        batch_attention_padding = pad_and_cat2d(l_attention, padding_value=0)[inv_offset] # [batch, max_action_num, seq_len]

        log_action_dist = log_action_prob.view(-1, 1) + safe_log(action_dist)

        if t == env.max_hop - 1:
            action, log_action_prob, attention_matrix, action_offset = top_k_answer_unique(log_action_dist, action_space, batch_attention_padding, batch_size, beam_size)
        else:
            action, log_action_prob, attention_matrix, action_offset = top_k_action(log_action_dist, action_space, batch_attention_padding, batch_size, beam_size)
        
        path_list, (h_t, c_t) = generator.update_path(action, path_hidden, offset = action_offset) 
        if args.use_coverage_attn:
            l_coverage, new_coverage = generator.update_coverage(attention_matrix, l_coverage, offset = action_offset)
            env.step(action, path_list, (h_t, c_t), l_coverage, new_coverage, attention_matrix)
        else:
            env.step(action, path_list, (h_t, c_t))
        
        rearrange_vector_list(l_log_action_probs, action_offset)
        l_log_action_probs.append(log_action_prob)
        adjust_search_trace(l_search_trace, action_offset)
        l_search_trace.append(action)
    
    return action, l_search_trace, l_log_action_probs