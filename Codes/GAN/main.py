import os
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from parse_args import args
from utils import build_qa_vocab, initialize_word_embedding, set_seed, process_text_file, load_all_triples_from_txt, get_adjacent, initialize_action_space, get_hops_mask, get_intermediate_path, calculate_intermediate_accuracy, sample_negative, get_dataset_path, get_id_vocab
from dataloader import Dataset_GAN, DataLoader_GAN
from environment import Environment
from policy_network import Policy_Network
from discriminator import Discriminator
from rollout import sample_action, rollout_beam, rollout_normal

flag_words = ['<pad>', '<unk>']
glove_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Dataset/glove.840B.300d.zip")

def pretrain_generator(args, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, ckpt_path):
    env_train = Environment(args, is_Pretrain_Gen = True, isTrain = True)
    env_valid = Environment(args, is_Pretrain_Gen = True, isTrain = False)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_generator_pretrain, weight_decay=args.weight_decay)
    best_dev_metrics = -float("inf")
    best_model = generator.state_dict()

    start = time.time()
    for epoch_id in range(0, args.gen_pretrain_epoch):
        generator.train()
        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")
        
        l_epoch_loss = []

        for i_batch, batch_data in enumerate(train_loader):
            if use_cuda:
                batch_data = [batch_item.cuda() for batch_item in batch_data]

            env_train.reset(batch_data, generator)
            
            batch_question, batch_question_len, batch_head, batch_relation_path, batch_relation_lengths, batch_intermediate_entities = env_train.return_batch_data()
            batch_size, _ = batch_question.shape

            batch_hops_mask = get_hops_mask(batch_relation_path, batch_relation_lengths)
            
            batch_nll_loss = 0.0

            for t in range(env_train.max_hop):
                if args.use_coverage_attn:
                    path_trace, path_hidden, l_coverage = env_train.observe()
                    coverage_t = l_coverage[-1] # [batch, seq_len]
                else:
                    path_trace, path_hidden = env_train.observe()
                    coverage_t = None

                last_r, e_t = path_trace[-1]
                golden_relation_t = batch_relation_path[:, t].view(-1, 1) # [batch, 1]
                golden_entity_t = batch_intermediate_entities[:, t].view(-1, 1) # [batch, 1]
                golden_action_t = (golden_relation_t, golden_entity_t)
                batch_hops_mask_t = batch_hops_mask[:, t]
                
                batch_path_hidden = path_hidden[-1][0][-1, :, :]
                db_outcomes, l_attention, l_golden_label, batch_nll_loss_t, inv_offset = generator.transit_pretrain(t, e_t, batch_question, batch_question_len, batch_path_hidden, d_entity2bucketid, d_action_space_buckets, last_r, coverage_t, env_train.max_hop, golden_action_t, batch_hops_mask_t, args.dataset, use_mask_trick = False)

                batch_nll_loss += batch_nll_loss_t

                action_sample, action_prob, attention_matrix = sample_action(db_outcomes, l_attention, inv_offset, l_golden_label) 

                path_list, (h_t, c_t) = generator.update_path(action_sample, path_hidden)
                if args.use_coverage_attn:
                    l_coverage, new_coverage = generator.update_coverage(attention_matrix, l_coverage)
                    env_train.step(action_sample, path_list, (h_t, c_t), l_coverage, new_coverage, attention_matrix)
                else:
                    env_train.step(action_sample, path_list, (h_t, c_t))
            
            if args.use_coverage_attn:
                batch_coverage_loss = env_train.return_coverage_loss(batch_hops_mask)
                batch_loss = batch_nll_loss + args.coverage_coefficient * batch_coverage_loss
            else:
                batch_loss = batch_nll_loss

            l_epoch_loss.append(1.0 * batch_loss.item() / batch_size)
            
            optimizer.zero_grad()
            batch_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(generator.parameters(), args.grad_norm)
            optimizer.step()

        # Check dev set performance
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            generator.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches") 
            total_hits1 = 0.0
            total_num = 0.0 
            total_relation_path_acc_num = 0.0
            total_intermediate_accuracy = torch.zeros(args.max_hop + 1).cuda()
            total_intermediate_num = torch.zeros(args.max_hop + 1).cuda()

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    if use_cuda:
                        batch_data = [batch_item.cuda() for batch_item in batch_data]
                    
                    env_valid.reset(batch_data, generator)
            
                    batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_valid.return_batch_data()
                    batch_size = batch_head.shape[0]
                    
                    action, l_search_trace, _ = rollout_beam(args, generator, env_valid, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, args.beam_size, use_mask_trick = False)
                    
                    batch_pred_e2 = action[1].view(batch_size, -1) # [batch, beam_size]
                    batch_pred_e2_top1 = batch_pred_e2[:, 0].view(batch_size, -1)
                    batch_hits1 = torch.sum(torch.gather(batch_answers, 1, batch_pred_e2_top1).view(-1)).item()
                    total_hits1 += batch_hits1
                    total_num += batch_size

                    batch_pred_relations = get_intermediate_path(batch_size, l_search_trace) # [batch, max_hop]
                    batch_relation_accuracy_sum, batch_hops_mask_sum, batch_relation_path_acc_num = calculate_intermediate_accuracy(batch_relation_path, batch_relation_lengths, batch_pred_relations)
                    total_relation_path_acc_num += batch_relation_path_acc_num

                    pad = nn.ConstantPad1d((0, args.max_hop + 1 - env_valid.max_hop), 0)
                    batch_relation_accuracy_sum = pad(batch_relation_accuracy_sum)
                    batch_hops_mask_sum = pad(batch_hops_mask_sum)

                    total_intermediate_accuracy += batch_relation_accuracy_sum
                    total_intermediate_num += batch_hops_mask_sum
                    
                answer_hits_1 = 1.0 * total_hits1 / total_num
                relation_path_acc = 1.0 * total_relation_path_acc_num / total_num
                intermediate_accuracy = 1.0 * total_intermediate_accuracy / total_intermediate_num
                intermediate_accuracy = torch.where(torch.isnan(intermediate_accuracy), torch.full_like(intermediate_accuracy, 0), intermediate_accuracy)

                # Save checkpoint
                if answer_hits_1 > best_dev_metrics:
                    best_dev_metrics = answer_hits_1
                    best_model = generator.state_dict()
                    print('Epoch {}: best vaild Hits@1 = {}. relation path accuracy = {}'.format(epoch_id, best_dev_metrics, relation_path_acc))
                    print(intermediate_accuracy)

            ckpt_path_new = ckpt_path[:-5] + "_{}.ckpt".format(epoch_id + 1)
            torch.save(best_model, ckpt_path_new)

def pretrain_discriminator(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, ckpt_path_discriminator, d_relation2id):
    env_train = Environment(args, is_Pretrain_Gen = False, isTrain = True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_discriminator_pretrain, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='sum')
    best_dev_metrics = -float("inf")
    best_model = discriminator.state_dict()

    for epoch_id in range(0, args.dis_pretrain_epoch):
        l_epoch_loss = []
        l_epoch_pos_scores = []
        l_epoch_neg_scores = []
        discriminator.train()
        train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")

        for i_batch, batch_data in enumerate(train_loader):
            if use_cuda:
                batch_data = [batch_item.cuda() for batch_item in batch_data]
            
            env_train.reset(batch_data, generator)

            batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_train.return_batch_data()
            batch_size = batch_head.shape[0]
            
            _, l_search_trace, _ = rollout_beam(args, generator, env_train, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, args.beam_size_dis, use_mask_trick = False)
        
            batch_pred_relations = get_intermediate_path(batch_size, l_search_trace, top1= False)
            batch_negative_samples, batch_negative_len = sample_negative(batch_pred_relations, batch_relation_path, batch_relation_lengths, d_relation2id)

            batch_neg_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_negative_samples, batch_negative_len) # [batch]
            batch_neg_loss = discriminator.calculate_loss(criterion, batch_neg_scores_sigmoid, label = 0)

            batch_pos_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_relation_path, batch_relation_lengths) 
            batch_pos_loss = discriminator.calculate_loss(criterion, batch_pos_scores_sigmoid, label = 1)

            batch_loss = batch_neg_loss + batch_pos_loss
            l_epoch_loss.append(batch_loss.item())
            l_epoch_pos_scores.extend(batch_pos_scores_sigmoid.cpu().detach().numpy().tolist())
            l_epoch_neg_scores.extend(batch_neg_scores_sigmoid.cpu().detach().numpy().tolist())

            optimizer.zero_grad()
            batch_loss.backward()
            if args.grad_norm > 0:
                clip_grad_norm_(discriminator.parameters(), args.grad_norm)
            optimizer.step()

        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            discriminator.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches") 
            l_pos_scores = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    if use_cuda:
                        batch_data = [batch_item.cuda() for batch_item in batch_data]
                    
                    batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = batch_data
                    
                    batch_pos_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_relation_path, batch_relation_lengths)
                    l_pos_scores.extend(batch_pos_scores_sigmoid.cpu().detach().numpy().tolist())
                
                pos_mean_scores = np.mean(l_pos_scores)
                
                # Save checkpoint
                if pos_mean_scores > best_dev_metrics:
                    best_dev_metrics = pos_mean_scores
                    best_model = discriminator.state_dict()
                    print('Epoch {}: best positive score = {}.'.format(epoch_id, best_dev_metrics))
            
    torch.save(best_model, ckpt_path_discriminator)

def generator_relation_path(env, NO_OP_RELATION = 2):
    if args.use_coverage_attn:
        path_trace, path_hidden, l_coverage = env.observe()
    else:
        path_trace, path_hidden = env.observe()
    
    relation_path_trace = [trace[0].unsqueeze(-1) for i, trace in enumerate(path_trace) if i > 0]
    relation_path_tensor = torch.cat(relation_path_trace, dim = -1) # [batch, max_hop]

    l_relation_path_len = []

    for batch_i in range(relation_path_tensor.shape[0]):
        relation_path_i = relation_path_tensor[batch_i]
        equal_self_loop = (relation_path_i == NO_OP_RELATION).long()
        all_self_loop_indice = torch.nonzero(equal_self_loop).view(-1)
        if len(all_self_loop_indice) == 0:
            path_len_i = env.max_hop
        else:
            path_len_i = all_self_loop_indice[0] + 1
            path_len_i = path_len_i.item()
        l_relation_path_len.append(path_len_i)
    
    batch_relation_path_len = torch.LongTensor(l_relation_path_len).cuda() # [batch * mc]
    return relation_path_tensor, batch_relation_path_len

def adversarial_training(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, d_relation2id, ckpt_path_generator, ckpt_path_discriminator):
    env_train = Environment(args, is_Pretrain_Gen = False, isTrain = True)
    env_valid = Environment(args, is_Pretrain_Gen = False, isTrain = False)
    
    optimizer_gen = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=args.lr_generator_adversarial, weight_decay=args.weight_decay)
    optimizer_dis = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr_discriminator_adversarial, weight_decay=args.weight_decay)
    
    criterion_dis = nn.BCELoss(reduction='sum')
    best_dev_metrics = -float("inf")
    best_relation_acc = -float("inf")
    best_epoch = -1
    best_gen_model = generator.state_dict()
    best_dis_model = discriminator.state_dict()

    for epoch_id in range(0, args.total_epoch):
        print("epoch = ", epoch_id)
        for d_num in range(args.D_steps):
            generator.eval()
            discriminator.train()
            l_epoch_loss = []
            l_epoch_pos_scores = []
            l_epoch_neg_scores = []
            train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")

            for i_batch, batch_data in enumerate(train_loader):
                if use_cuda:
                    batch_data = [batch_item.cuda() for batch_item in batch_data]
                
                env_train.reset(batch_data, generator)

                batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_train.return_batch_data()
                batch_size = batch_head.shape[0]
                
                _, l_search_trace, _ = rollout_beam(args, generator, env_train, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, args.beam_size_dis, use_mask_trick = False)
            
                batch_pred_relations = get_intermediate_path(batch_size, l_search_trace, top1= False)
                batch_negative_samples, batch_negative_len = sample_negative(batch_pred_relations, batch_relation_path, batch_relation_lengths, d_relation2id)

                batch_neg_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_negative_samples, batch_negative_len) # [batch]
                batch_neg_loss = discriminator.calculate_loss(criterion_dis, batch_neg_scores_sigmoid, label = 0)

                batch_pos_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_relation_path, batch_relation_lengths) 
                batch_pos_loss = discriminator.calculate_loss(criterion_dis, batch_pos_scores_sigmoid, label = 1)

                batch_loss = batch_neg_loss + batch_pos_loss
                l_epoch_loss.append(batch_loss.item())
                l_epoch_pos_scores.extend(batch_pos_scores_sigmoid.cpu().detach().numpy().tolist())
                l_epoch_neg_scores.extend(batch_neg_scores_sigmoid.cpu().detach().numpy().tolist())

                optimizer_dis.zero_grad()
                batch_loss.backward()
                if args.grad_norm > 0:
                    clip_grad_norm_(discriminator.parameters(), args.grad_norm)
                optimizer_dis.step()
        
        for g_num in range(args.G_steps):
            generator.train()
            discriminator.eval()
            l_epoch_loss = []
            train_loader = tqdm(train_dataloader, total=len(train_dataloader), unit="batches")

            for i_batch, batch_data in enumerate(train_loader):
                if use_cuda:
                    batch_data = [batch_item.cuda() for batch_item in batch_data]
                
                env_train.reset(batch_data, generator)
                batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_train.return_batch_data()
                batch_size, _ = batch_question.shape

                l_log_action_probs, l_action_value = rollout_normal(args, env_train, generator, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, end = env_train.max_hop, use_mask_trick = False)

                batch_pred_e2 = env_train.get_pred_entities()
                binary_rewards = torch.gather(batch_answers, 1, batch_pred_e2.unsqueeze(-1)).squeeze(-1).float() # [batch]
                
                if args.reward_calculation_type == 2 or args.reward_calculation_type == 4:
                    relation_path_tensor, batch_relation_path_len = generator_relation_path(env_train) # [batch, max-hop], [batch]
                    with torch.no_grad():
                        batch_mc_scores_T = discriminator(batch_question, batch_question_len, relation_path_tensor, batch_relation_path_len) # [batch]

                cum_rewards = [0] * env_train.max_hop
                R = torch.zeros(batch_size).cuda()
                policy_loss = torch.zeros(batch_size).cuda()
                value_loss = torch.zeros(batch_size).cuda()

                if args.use_gae:
                    gae = torch.zeros(batch_size).cuda()
                    l_action_value.append(R)

                if args.reward_calculation_type == 2:
                    cum_rewards[-1] = binary_rewards * args.reward_coefficient + (1 - args.reward_coefficient) * batch_mc_scores_T
                elif args.reward_calculation_type == 3:
                    cum_rewards[-1] = binary_rewards
                elif args.reward_calculation_type == 4:
                    cum_rewards[-1] = batch_mc_scores_T
                    
                for i in reversed(range(env_train.max_hop)):
                    R = args.gamma * R + cum_rewards[i]
                    total_R = R

                    if args.use_actor_critic:
                        advantage = total_R - l_action_value[i]
                        value_loss = value_loss + 0.5 * advantage.pow(2)

                        policy_loss = policy_loss - advantage.detach() * l_log_action_probs[i]

                        if args.use_gae:
                            delta_t = total_R + args.gamma * l_action_value[i+1] - l_action_value[i]
                            gae = gae * args.gamma * args.tau + delta_t
                            policy_loss = policy_loss - gae.detach() * l_log_action_probs[i]

                    else:
                        policy_loss = policy_loss - total_R.detach() * l_log_action_probs[i]
                    
            
                if args.use_actor_critic:
                    rl_loss = policy_loss + args.value_loss_coef * value_loss
                else:
                    rl_loss = policy_loss

                rl_loss_mean = rl_loss.mean()
                l_epoch_loss.append(rl_loss_mean.item())

                optimizer_gen.zero_grad()
                rl_loss_mean.backward()
                if args.grad_norm > 0:
                    clip_grad_norm_(generator.parameters(), args.grad_norm)
                optimizer_gen.step()
            
        # Check dev set performance
        if epoch_id % args.num_wait_epochs == args.num_wait_epochs - 1:
            generator.eval()
            discriminator.eval()
            valid_loader = tqdm(valid_dataloader, total=len(valid_dataloader), unit="batches") 
            total_hits1 = 0.0
            total_num = 0.0 
            total_relation_path_acc_num = 0.0
            total_intermediate_accuracy = torch.zeros(args.max_hop + 1).cuda()
            total_intermediate_num = torch.zeros(args.max_hop + 1).cuda()
            l_pos_scores = []

            with torch.no_grad():
                for i_batch, batch_data in enumerate(valid_loader):
                    if use_cuda:
                        batch_data = [batch_item.cuda() for batch_item in batch_data]
                    
                    env_valid.reset(batch_data, generator)
            
                    batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_valid.return_batch_data()
                    batch_size = batch_head.shape[0]
                    
                    action, l_search_trace, _ = rollout_beam(args, generator, env_valid, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, args.beam_size, use_mask_trick = False)
                    
                    batch_pred_e2 = action[1].view(batch_size, -1) # [batch, beam_size]
                    batch_pred_e2_top1 = batch_pred_e2[:, 0].view(batch_size, -1)
                    batch_hits1 = torch.sum(torch.gather(batch_answers, 1, batch_pred_e2_top1).view(-1)).item()
                    total_hits1 += batch_hits1
                    total_num += batch_size

                    batch_pred_relations = get_intermediate_path(batch_size, l_search_trace) # [batch, max_hop]
                    batch_relation_accuracy_sum, batch_hops_mask_sum, batch_relation_path_acc_num = calculate_intermediate_accuracy(batch_relation_path, batch_relation_lengths, batch_pred_relations)
                    total_relation_path_acc_num += batch_relation_path_acc_num

                    pad = nn.ConstantPad1d((0, args.max_hop + 1 - env_valid.max_hop), 0)
                    batch_relation_accuracy_sum = pad(batch_relation_accuracy_sum)
                    batch_hops_mask_sum = pad(batch_hops_mask_sum)

                    total_intermediate_accuracy += batch_relation_accuracy_sum
                    total_intermediate_num += batch_hops_mask_sum

                    batch_pos_scores_sigmoid = discriminator(batch_question, batch_question_len, batch_relation_path, batch_relation_lengths)
                    l_pos_scores.extend(batch_pos_scores_sigmoid.cpu().detach().numpy().tolist())
                    
    
                answer_hits_1 = 1.0 * total_hits1 / total_num
                relation_path_acc = 1.0 * total_relation_path_acc_num / total_num
                intermediate_accuracy = 1.0 * total_intermediate_accuracy / total_intermediate_num
                intermediate_accuracy = torch.where(torch.isnan(intermediate_accuracy), torch.full_like(intermediate_accuracy, 0), intermediate_accuracy)
                
                # Save checkpoint
                if answer_hits_1 > best_dev_metrics:
                    best_epoch = epoch_id + 1
                    best_dev_metrics = answer_hits_1
                    best_relation_acc = relation_path_acc
                    best_gen_model = generator.state_dict()
                    best_dis_model = discriminator.state_dict()
                    torch.save(best_gen_model, ckpt_path_generator)
                    torch.save(best_dis_model, ckpt_path_discriminator)
                    print('Epoch {}: best vaild Hits@1 = {}. relation path accuracy = {}'.format(epoch_id, best_dev_metrics, relation_path_acc))

    return best_epoch, best_dev_metrics, best_relation_acc

def run_inference(args, generator, test_dataloader, ckpt_path_generator, d_entity2bucketid, d_action_space_buckets, d_relation2id, d_entity2id):
    generator.eval()
    generator.load(ckpt_path_generator)

    env_test = Environment(args, is_Pretrain_Gen = False, isTrain = False)
    test_loader = tqdm(test_dataloader, total=len(test_dataloader), unit="batches")
    total_hits1 = 0.0
    total_num = 0.0 
    total_relation_path_acc_num = 0.0
    total_intermediate_accuracy = torch.zeros(args.max_hop + 1).cuda()
    total_intermediate_num = torch.zeros(args.max_hop + 1).cuda()

    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            if use_cuda:
                batch_data = [batch_item.cuda() for batch_item in batch_data]
            
            env_test.reset(batch_data, generator)
    
            batch_question, batch_question_len, batch_head, batch_answers, batch_relation_path, batch_relation_lengths = env_test.return_batch_data()
            batch_size = batch_head.shape[0]
            
            action, l_search_trace, l_log_action_probs = rollout_beam(args, generator, env_test, batch_size, batch_head, batch_question, batch_question_len, d_entity2bucketid, d_action_space_buckets, args.beam_size, use_mask_trick = False)
            
            batch_pred_e2 = action[1].view(batch_size, -1) # [batch, beam_size]
            batch_pred_e2_top1 = batch_pred_e2[:, 0].view(batch_size, -1)
            batch_pred_results = torch.gather(batch_answers, 1, batch_pred_e2_top1).view(-1)
            batch_hits1 = torch.sum(batch_pred_results).item()
            total_hits1 += batch_hits1
            total_num += batch_size

            batch_pred_relations = get_intermediate_path(batch_size, l_search_trace) # [batch, max_hop]
            batch_relation_accuracy_sum, batch_hops_mask_sum, batch_relation_path_acc_num = calculate_intermediate_accuracy(batch_relation_path, batch_relation_lengths, batch_pred_relations)
            total_relation_path_acc_num += batch_relation_path_acc_num

            pad = nn.ConstantPad1d((0, args.max_hop + 1 - env_test.max_hop), 0)
            batch_relation_accuracy_sum = pad(batch_relation_accuracy_sum)
            batch_hops_mask_sum = pad(batch_hops_mask_sum)

            total_intermediate_accuracy += batch_relation_accuracy_sum
            total_intermediate_num += batch_hops_mask_sum

        answer_hits_1 = 1.0 * total_hits1 / total_num
        relation_path_acc = 1.0 * total_relation_path_acc_num / total_num
        intermediate_accuracy = 1.0 * total_intermediate_accuracy / total_intermediate_num
        intermediate_accuracy = torch.where(torch.isnan(intermediate_accuracy), torch.full_like(intermediate_accuracy, 0), intermediate_accuracy)

        return answer_hits_1, relation_path_acc, intermediate_accuracy

def run_train(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, d_relation2id, output_path):
    adversarial_path = os.path.join(output_path, "adversarial_learning")
    if not os.path.exists(adversarial_path):
        os.makedirs(adversarial_path)
        
    ckpt_path_generator = os.path.join(adversarial_path, "generator_GAN.ckpt")
    ckpt_path_discriminator = os.path.join(adversarial_path, "discriminator_GAN.ckpt")
    
    pretrain_gen_path = os.path.join(output_path, "pretrain_generator")
    pretrain_dis_path = os.path.join(output_path, "pretrain_discriminator")

    generator_pretrain_ckpt = os.path.join(pretrain_gen_path, "generator_pretrain_{}.ckpt".format(args.G_pretrain_step)) 
    discriminator_pretrain_ckpt = os.path.join(pretrain_dis_path, "discriminator_pretrain_{}.ckpt".format(args.G_pretrain_step)) 

    generator.load(generator_pretrain_ckpt)
    discriminator.load(discriminator_pretrain_ckpt)

    adversarial_training(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, d_relation2id, ckpt_path_generator, ckpt_path_discriminator)

if __name__ == "__main__":
    # 1. Set device & seed
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    set_seed(args.seed)
    
    if args.train:
        print(args.dataset)

        if args.dataset.startswith("MetaQA"):
            args.add_reversed_edges = True
            args.beam_size_dis = 1
        else:
            args.add_reversed_edges = False
            args.beam_size_dis = 3
        
        # 2. Set dataset path
        train_path_multi, train_path_single, valid_path, test_path, kb_path, entity2id_path, relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, output_path = get_dataset_path(args)

        d_entity2id, d_relation2id = get_id_vocab(args, kb_path,entity2id_path, relation2id_path)

        if not os.path.isfile(word2id_path):
            d_word2id = build_qa_vocab(train_path_multi, valid_path, word2id_path, args.min_freq, flag_words)
        else:
            d_word2id = pickle.load(open(word2id_path, 'rb'))

        if not os.path.isfile(word_embedding_path):
            word_embeddings = initialize_word_embedding(d_word2id, glove_path, word_embedding_path)
        else:
            word_embeddings = np.load(word_embedding_path)
        word_embeddings = torch.from_numpy(word_embeddings)

        if os.path.isfile(entity_embedding_path):
            entity_embeddings = np.load(entity_embedding_path)
            entity_embeddings = torch.from_numpy(entity_embeddings)
        
        if os.path.isfile(relation_embedding_path):
            relation_embeddings = np.load(relation_embedding_path)
            relation_embeddings = torch.from_numpy(relation_embeddings)

        if use_cuda:
            word_embeddings = word_embeddings.cuda()
            entity_embeddings = entity_embeddings.cuda()
            relation_embeddings = relation_embeddings.cuda()
        
        triples = load_all_triples_from_txt(kb_path, d_entity2id, d_relation2id, args.add_reversed_edges)
        triple_dict = get_adjacent(triples)
        d_entity2bucketid, d_action_space_buckets = initialize_action_space(len(d_entity2id), triple_dict, args.bucket_interval) 

        l_pretrain_data = process_text_file(train_path_single, d_relation2id, d_entity2id, d_word2id, isTrain = True, is_Pretrain_gen= True)
        l_train_data  = process_text_file(train_path_multi, d_relation2id, d_entity2id, d_word2id, isTrain = True, is_Pretrain_gen = False)
        l_valid_data  = process_text_file(valid_path, d_relation2id, d_entity2id, d_word2id, isTrain = False, is_Pretrain_gen = False)
        l_test_data  = process_text_file(test_path, d_relation2id, d_entity2id, d_word2id, isTrain = False, is_Pretrain_gen = False)

        pretrain_dataset = Dataset_GAN(l_pretrain_data, len(d_entity2id), isTrain=True, is_Pretrain_gen=True)
        pretrain_dataloader = DataLoader_GAN(pretrain_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

        train_dataset = Dataset_GAN(l_train_data, len(d_entity2id), isTrain=True, is_Pretrain_gen= False)
        train_dataloader = DataLoader_GAN(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        
        valid_dataset = Dataset_GAN(l_valid_data, len(d_entity2id), isTrain=False, is_Pretrain_gen= False)
        valid_dataloader = DataLoader_GAN(valid_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        
        pretrain_gen_path = os.path.join(output_path, "pretrain_generator")
        if not os.path.exists(pretrain_gen_path):
            os.mkdir(pretrain_gen_path)
        
        pretrain_dis_path = os.path.join(output_path, "pretrain_discriminator")
        if not os.path.exists(pretrain_dis_path):
            os.mkdir(pretrain_dis_path)
        
        generator = Policy_Network(args, word_embeddings, entity_embeddings, relation_embeddings)

        discriminator = Discriminator(args, word_embeddings, relation_embeddings)

        if use_cuda:
            generator = generator.cuda()
            discriminator = discriminator.cuda()
        
        if len(os.listdir(pretrain_gen_path)) == 0:
            generator_pretrain_ckpt = os.path.join(pretrain_gen_path, "generator_pretrain.ckpt")
            pretrain_generator(args, generator, pretrain_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, generator_pretrain_ckpt)
        
        if len(os.listdir(pretrain_dis_path)) == 0:
            for gen_ckpt in os.listdir(pretrain_gen_path):
                if not gen_ckpt.endswith("ckpt"):
                    continue
                pretrain_epoch = gen_ckpt.split("_")[-1]
                discriminator_pretrain_ckpt = os.path.join(pretrain_dis_path, "discriminator_pretrain_" + pretrain_epoch)

                generator.load(os.path.join(pretrain_gen_path, gen_ckpt))
                generator.eval()

                pretrain_discriminator(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, discriminator_pretrain_ckpt, d_relation2id)
        
        run_train(args, discriminator, generator, train_dataloader, valid_dataloader, d_entity2bucketid, d_action_space_buckets, d_relation2id, output_path)

    elif args.eval:
        if args.dataset.startswith("MetaQA"):
            args.add_reversed_edges = True
        else:
            args.add_reversed_edges = False
        
        # 2. Set dataset path
        train_path_multi, train_path_single, valid_path, test_path, kb_path, entity2id_path, relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, output_path = get_dataset_path(args)

        d_entity2id, d_relation2id = get_id_vocab(args, kb_path,entity2id_path, relation2id_path)

        if not os.path.isfile(word2id_path):
            d_word2id = build_qa_vocab(train_path_multi, valid_path, word2id_path, args.min_freq, flag_words)
        else:
            d_word2id = pickle.load(open(word2id_path, 'rb'))

        if not os.path.isfile(word_embedding_path):
            word_embeddings = initialize_word_embedding(d_word2id, glove_path, word_embedding_path)
        else:
            word_embeddings = np.load(word_embedding_path)
        word_embeddings = torch.from_numpy(word_embeddings)

        if os.path.isfile(entity_embedding_path):
            entity_embeddings = np.load(entity_embedding_path)
            entity_embeddings = torch.from_numpy(entity_embeddings)
        
        if os.path.isfile(relation_embedding_path):
            relation_embeddings = np.load(relation_embedding_path)
            relation_embeddings = torch.from_numpy(relation_embeddings)

        if use_cuda:
            word_embeddings = word_embeddings.cuda()
            entity_embeddings = entity_embeddings.cuda()
            relation_embeddings = relation_embeddings.cuda()
        
        triples = load_all_triples_from_txt(kb_path, d_entity2id, d_relation2id, args.add_reversed_edges)
        triple_dict = get_adjacent(triples)
        d_entity2bucketid, d_action_space_buckets = initialize_action_space(len(d_entity2id), triple_dict, args.bucket_interval)
        l_test_data = process_text_file(test_path, d_relation2id, d_entity2id, d_word2id, isTrain = False, is_Pretrain_gen = False)

        test_dataset = Dataset_GAN(l_test_data, len(d_entity2id), isTrain=False, is_Pretrain_gen= False)
        test_dataloader = DataLoader_GAN(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

        generator = Policy_Network(args, word_embeddings, entity_embeddings, relation_embeddings)

        if use_cuda:
            generator = generator.cuda()

        ckpt_path_generator = os.path.join(output_path, "adversarial_learning/best_generator_GAN.ckpt") 

        test_hits1, test_relation_acc, test_intermediate_accuracy = run_inference(args, generator, test_dataloader, ckpt_path_generator, d_entity2bucketid, d_action_space_buckets, d_relation2id, d_entity2id)
        print(args.dataset, args.beam_size, test_hits1)