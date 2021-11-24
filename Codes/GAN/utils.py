import re
import os
import copy
import random
import torch
import pickle
import zipfile
import numpy as np
from collections import Counter
from collections import defaultdict
import operator
from tqdm import tqdm
import torch.nn as nn

EPSILON = float(np.finfo(float).eps)

def safe_log(x):
    return torch.log(x + EPSILON)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_er2id(kb_file, e_output_file, r_output_file, pre_entities,pre_relations, reversed_edge = False):
    l_entities = []
    l_relations = []

    with open(kb_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            s, r, o = line.split("\t")
            s, r, o = s.strip(), r.strip(), o.strip()
            l_entities.append(s)
            l_relations.append(r)
            if reversed_edge == True:
                l_relations.append(r + "_inverse")
            l_entities.append(o)
    
    l_entities = list(set(l_entities))
    l_entities = pre_entities + l_entities

    l_relations = list(set(l_relations))
    l_relations = pre_relations + l_relations

    with open(e_output_file, "w") as f:
        for i , e in enumerate(l_entities):
            f.write(e + "\t" + str(i) + "\n")

    with open(r_output_file, "w") as f:
        for i , r in enumerate(l_relations):
            f.write(r + "\t" + str(i) + "\n")

def read_vocab(vocab_file):
    d_item2id = {}

    with open(vocab_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            d_item2id[items[0]] = int(items[1])

    return d_item2id

def index2word(word2id):
    return {i: w for w, i in word2id.items()}

# Flatten and pack nested lists using recursion
def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def load_all_triples_from_txt(data_path, d_entity2id, d_relation2id, add_reverse_relations=False):
    triples = []
    
    def triple2ids(s, r, o):
        return (d_entity2id[s], d_relation2id[r], d_entity2id[o])
    
    with open(data_path) as f:
        for line in f.readlines():
            s, r, o = line.strip().split("\t")
            s, r, o = s.strip(), r.strip(), o.strip()
            triples.append(triple2ids(s, r, o))
            if add_reverse_relations:
                triples.append(triple2ids(o, r + '_inverse', s))
    
    # print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def get_adjacent(triples):
    triple_dict = defaultdict(defaultdict)

    for triple in triples:
        s_id, r_id, o_id = triple
        
        if r_id not in triple_dict[s_id]:
            triple_dict[s_id][r_id] = set()
        triple_dict[s_id][r_id].add(o_id)

    return triple_dict


def print_all_model_parameters(model):
    # print('\nModel Parameters')
    # print('--------------------------')
    # for name, param in model.named_parameters():
    #     print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
    param_sizes = [param.numel() for param in model.parameters()]
    print('Total # parameters = {}'.format(sum(param_sizes)))
    # print('--------------------------')


def build_qa_vocab(train_file, valid_file, word2id_output_file, min_freq, flag_words):
    count = Counter()
    with open(train_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            question_pattern = items[0].strip()
            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)
    
    with open(valid_file) as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            items = line.split("\t")
            question_pattern = items[0].strip()
            words_pattern = [word for word in question_pattern.split(" ")]
            count.update(words_pattern)

    count = {k: v for k, v in count.items()}
    count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    vocab = [w[0] for w in count if w[1] >= min_freq]
    vocab = flag_words + vocab
    word2id = {k: v for k, v in zip(vocab, range(0, len(vocab)))}
    print("word len: ", len(word2id))
    assert word2id['<pad>'] == 0, "ValueError: '<pad>' id is not 0"

    with open(word2id_output_file, 'wb') as fw:
        pickle.dump(word2id, fw)

    return word2id

def initialize_word_embedding(word2id, glove_path, word_embedding_file):
    word_embeddings = np.random.uniform(-0.1, 0.1, (len(word2id), 300))
    seen_words = []

    gloves = zipfile.ZipFile(glove_path)
    for glove in gloves.infolist():
        with gloves.open(glove) as f:
            for line in f:
                if line != "":
                    splitline = line.split()
                    word = splitline[0].decode('utf-8')
                    embedding = splitline[1:]
                    if word in word2id and len(embedding) == 300:
                        temp = np.array([float(val) for val in embedding])
                        word_embeddings[word2id[word], :] = temp/np.sqrt(np.sum(temp**2))
                        seen_words.append(word)

    word_embeddings = word_embeddings.astype(np.float32)
    word_embeddings[0, :] = 0. # pad初始化成0.
    print("pretrained vocab %s among %s" %(len(seen_words), len(word_embeddings)))
    unseen_words = set([k for k in word2id]) - set(seen_words)
    print("unseen words = ", len(unseen_words), unseen_words) 
    np.save(word_embedding_file, word_embeddings)
    return word_embeddings

def token_to_id(token, token2id, flag_words = "<unk>"):
    return token2id[token] if token in token2id else token2id[flag_words]


def process_text_file(text_file, d_relation2id, d_entity2id, d_word2id, isTrain, is_Pretrain_gen, NO_OP_RELATION = 2): 
    l_data = []
    if isTrain and is_Pretrain_gen:

        with open(text_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                question_pattern, topic_entity, answer_entities, relation_path, full_path = line.split('\t')
                answer_entities = answer_entities.split("|")
                relations = relation_path.split("|")
                items = full_path.split("##")

                question_pattern_id = [token_to_id(word, d_word2id) for word in question_pattern.strip().split(" ")]
                topic_entity_id = d_entity2id[topic_entity]
                answer_entities_id = [d_entity2id[entity] for entity in answer_entities]
                relations_id = [ d_relation2id[relation] for relation in relations] + [NO_OP_RELATION] 
                intermediate_entities_id = [d_entity2id[e] for i,e in enumerate(items) if i > 0 and i % 2 == 0]
                intermediate_entities_id = intermediate_entities_id + [intermediate_entities_id[-1]]

                l_data.append([topic_entity_id, question_pattern_id, answer_entities_id, relations_id, intermediate_entities_id])
    else: 
        with open(text_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                question_pattern, topic_entity, answer_entities, relation_path = line.split('\t')
                answer_entities = answer_entities.split("|")
                relations = relation_path.split("|")

                question_pattern_id = [token_to_id(word, d_word2id) for word in question_pattern.strip().split(" ")]
                topic_entity_id = d_entity2id[topic_entity]
                answer_entities_id = [d_entity2id[entity] for entity in answer_entities]
                relations_id = [d_relation2id[relation] for relation in relations] + [NO_OP_RELATION]

                l_data.append([topic_entity_id, question_pattern_id, answer_entities_id, relations_id])
     
    return l_data


def getNeighbourhoodbyDict(entities, adj_list, hop, dataset, center = True):
    l_neighbourhood = []
    if type(entities) is not list:
        l_entities = [entities]
    else:
        l_entities = entities

    while(hop > 0):
        hop -= 1
        l_one_step = []
        for entity in l_entities:
            for edge in adj_list[entity]:
                l_one_step.extend([i for i in list(adj_list[entity][edge])])
                l_neighbourhood.extend([i for i in list(adj_list[entity][edge])])
        l_entities.clear()
        l_entities.extend([i for i in l_one_step])
    
    if re.search("\dH$", dataset):
        l_neighbourhood = list(set(l_entities))
    else: 
        l_neighbourhood = list(set(l_neighbourhood))
    
    if center == False:
        if entities in l_neighbourhood:
            l_neighbourhood.remove(entities)

    return l_neighbourhood

def get_all_entity_neighours(valid_data, test_data, triple_dict, args):
    d_entity_neighours = {}
    all_data = valid_data + test_data
    
    all_data_tqdm = tqdm(all_data, total=len(all_data), unit="data")

    for i, row in enumerate(all_data_tqdm):
        topic_entity_id, _, _ = row
        if topic_entity_id not in d_entity_neighours:
            l_neighbourhood = getNeighbourhoodbyDict(topic_entity_id, triple_dict, args.max_hop, args.dataset)
            d_entity_neighours[topic_entity_id] = l_neighbourhood
        else:
            continue

    return d_entity_neighours
    
def getEntityActions(subject, triple_dict, NO_OP_RELATION = 2):
    action_space = []

    if subject in triple_dict:
        for relation in triple_dict[subject]:
            objects = triple_dict[subject][relation]
            for obj in objects: 
                action_space.append((relation, obj))
        
    action_space.insert(0, (NO_OP_RELATION, subject)) 

    return action_space

def vectorize_action_space(action_space_list, action_space_size, DUMMY_ENTITY = 0, DUMMY_RELATION = 0):
    bucket_size = len(action_space_list)
    r_space = torch.zeros(bucket_size, action_space_size) + DUMMY_ENTITY 
    e_space = torch.zeros(bucket_size, action_space_size) + DUMMY_RELATION
    r_space = r_space.long()
    e_space = e_space.long()
    action_mask = torch.zeros(bucket_size, action_space_size)
    for i, action_space in enumerate(action_space_list): 
        for j, (r, e) in enumerate(action_space):
            r_space[i, j] = r
            e_space[i, j] = e
            action_mask[i, j] = 1
    action_mask = action_mask.long()
    return (r_space, e_space), action_mask

def initialize_action_space(num_entities, triple_dict, bucket_interval):
    d_action_space_buckets = {}
    d_action_space_buckets_discrete = defaultdict(list)
    d_entity2bucketid = torch.zeros(num_entities, 2).long()
    num_facts_saved_in_action_table = 0

    for e1 in range(num_entities):
        action_space = getEntityActions(e1, triple_dict) # list
        key = int(len(action_space) / bucket_interval) + 1 
        d_entity2bucketid[e1, 0] = key
        d_entity2bucketid[e1, 1] = len(d_action_space_buckets_discrete[key])
        d_action_space_buckets_discrete[key].append(action_space)
        num_facts_saved_in_action_table += len(action_space)
    
    print('{} facts saved in action table'.format(
        num_facts_saved_in_action_table - num_entities)) 
    for key in d_action_space_buckets_discrete: 
        d_action_space_buckets[key] = vectorize_action_space(
            d_action_space_buckets_discrete[key], key * bucket_interval)
    
    return d_entity2bucketid, d_action_space_buckets


def pad_and_cat_action_space(action_spaces, inv_offset):
        db_r_space, db_e_space, db_action_mask = [], [], []
        for (r_space, e_space), action_mask in action_spaces:
            db_r_space.append(r_space)
            db_e_space.append(e_space)
            db_action_mask.append(action_mask)
        r_space = pad_and_cat(db_r_space, padding_value=0)[inv_offset]
        e_space = pad_and_cat(db_e_space, padding_value=0)[inv_offset]
        action_mask = pad_and_cat(db_action_mask, padding_value=0)[inv_offset]
        action_space = ((r_space, e_space), action_mask)
        return action_space
    
def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()

def pad_and_cat2d(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad2d((0, 0, 0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()

def pad_and_cat2d(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad2d((0, 0, 0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0).cuda()

def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]

def adjust_search_trace(search_trace, action_offset):
    for i, (r, e) in enumerate(search_trace):
        new_r = r[action_offset]
        new_e = e[action_offset]
        search_trace[i] = (new_r, new_e)

def get_dataset_path(args):
    if args.dataset.startswith("MetaQA"):
        dataset_name = "MetaQA"
    
        if args.dataset.endswith("1H"):
            args.max_hop = 1
            dataset_file = "MetaQA-1H"
        
        elif args.dataset.endswith("2H"):
            args.max_hop = 2
            dataset_file = "MetaQA-2H"
        
        elif args.dataset.endswith("3H"):
            args.max_hop = 3
            dataset_file = "MetaQA-3H"
        
        elif args.dataset.endswith("mix"):
            args.max_hop = 3
            dataset_file = "MetaQA-Mix"

        kb_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/kb.txt".format(dataset_name)
        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/entity2id.txt".format(dataset_name)
        relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/relation2id.txt".format(dataset_name)
        entity_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/entity_embeddings_ConvE.npy".format(dataset_name)
        relation_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/relation_embeddings_ConvE.npy".format(dataset_name)

    elif args.dataset.startswith("PQ"):
        dataset_name = "PathQuestion"
        if args.dataset == "PQ-2H":
            args.max_hop = 2
            dataset_file = "PQ-2H"
    
        elif args.dataset == "PQ-3H":
            args.max_hop = 3
            dataset_file = "PQ-3H"
        
        elif args.dataset == "PQ-mix":
            args.max_hop = 3
            dataset_file = "PQ-Mix"
    
        elif args.dataset == "PQL-2H":
            args.max_hop = 2
            dataset_file = "PQL-2H"
        
        elif args.dataset == "PQL-3H":
            args.max_hop = 3
            dataset_file = "PQL-3H"
        
        elif args.dataset == "PQL-mix":
            args.max_hop = 3
            dataset_file = "PQL-Mix"
        
        
        kb_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/kb.txt".format(dataset_name, dataset_file)
        entity2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/entity2id.txt".format(dataset_name, dataset_file)
        relation2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/relation2id.txt".format(dataset_name, dataset_file)
        entity_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/entity_embeddings_ConvE.npy".format(dataset_name, dataset_file)
        relation_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/relation_embeddings_ConvE.npy".format(dataset_name, dataset_file)

    train_path_multi = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/qa_train_multi.txt".format(dataset_name, dataset_file)
    train_path_single = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/qa_train_single.txt".format(dataset_name, dataset_file) 
    valid_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/qa_dev_multi.txt".format(dataset_name, dataset_file)
    test_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/Datasets/{}/{}/qa_test_multi.txt".format(dataset_name, dataset_file)
    
    word2id_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Datasets/{}/{}/word2id.pkl").format(dataset_name, dataset_file)
    word_embedding_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Datasets/{}/{}/word_embeddings.npy").format(dataset_name, dataset_file)
    output_path = os.path.abspath(os.path.join(os.getcwd(), "../..") + "/Outputs/{}/").format(dataset_file)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return train_path_multi, train_path_single, valid_path, test_path, kb_path, entity2id_path, relation2id_path, word2id_path, word_embedding_path, entity_embedding_path, relation_embedding_path, output_path

def get_id_vocab(args, kb_path, entity2id_path, relation2id_path):
    START_RELATION = 'START_RELATION'
    NO_OP_RELATION = 'NO_OP_RELATION'
    NO_OP_ENTITY = 'NO_OP_ENTITY'
    DUMMY_RELATION = 'DUMMY_RELATION'
    DUMMY_ENTITY = 'DUMMY_ENTITY'

    PADDING_ENTITIES = [DUMMY_ENTITY, NO_OP_ENTITY]
    PADDING_ENTITIES_ID = [0, 1]
    PADDING_RELATIONS = [DUMMY_RELATION, START_RELATION, NO_OP_RELATION]
    if not os.path.isfile(entity2id_path):
        if args.add_reversed_edges:
            build_er2id(kb_path, entity2id_path, relation2id_path, PADDING_ENTITIES, PADDING_RELATIONS, reversed_edge=True)
        else:
            build_er2id(kb_path, entity2id_path, relation2id_path, PADDING_ENTITIES, PADDING_RELATIONS)
    
    d_entity2id = read_vocab(entity2id_path)
    d_relation2id = read_vocab(relation2id_path)

    return d_entity2id, d_relation2id

def get_hops_mask(batch_relation_path, batch_relation_lengths):
    batch_size, max_hop = batch_relation_path.size()
    batch_hops_mask = batch_relation_path.new_zeros((batch_size, max_hop))
    for i in range(batch_size):
        sent_len = batch_relation_lengths[i]
        batch_hops_mask[i][:sent_len] = 1
    
    return batch_hops_mask

def get_intermediate_path(batch_size, l_search_trace, top1 = True):
    l_relations = [] 
    if top1:
        for trace_t in l_search_trace:
            batch_relation = trace_t[0].view(batch_size, -1) 
            l_relations.append(batch_relation[:, 0].unsqueeze(1)) # [batch_size, 1]
        
        batch_relations = torch.cat(l_relations, dim = 1).cuda() # [batch_size, max_hop]
    else:
        for trace_t in l_search_trace:
            batch_relation = trace_t[0].view(batch_size, -1) # [batch_size, k]
            l_relations.append(batch_relation.unsqueeze(2)) # [batch_size, k, 1]
        
        batch_relations = torch.cat(l_relations, dim = 2).cuda() # [batch_size, k, max_hop]

    return batch_relations
        
def calculate_intermediate_accuracy(batch_relation_path, batch_relation_lengths, batch_pred_relations):
    batch_relation_lengths = batch_relation_lengths - 1
    batch_hops_mask = get_hops_mask(batch_relation_path, batch_relation_lengths)
    batch_relation_accuracy = (batch_pred_relations == batch_relation_path).long()
    batch_relation_accuracy = batch_relation_accuracy * batch_hops_mask
    batch_relation_accuracy_sum = torch.sum(batch_relation_accuracy, dim = 0) # [max_hop]
    batch_hops_mask_sum = torch.sum(batch_hops_mask, dim = 0) # [max_hop]
    batch_relation_path_acc = torch.sum(batch_relation_accuracy, dim = 1) 
    batch_relation_path_acc_num = torch.sum((batch_relation_path_acc == batch_relation_lengths).long())
    return batch_relation_accuracy_sum, batch_hops_mask_sum, batch_relation_path_acc_num.item()

def sample_negative(batch_pred_relations, batch_relation_path, batch_relation_lengths, d_relation2id, NO_OP_RELATION = 2):
    batch_hops_mask = get_hops_mask(batch_relation_path, batch_relation_lengths)
    batch_size, k, max_hop = batch_pred_relations.shape

    l_negative_samples = []
    l_negative_len = []

    for batch_i in range(batch_size):
        golden_relation_path = batch_relation_path[batch_i].unsqueeze(0)
        golden_relation_len = batch_relation_lengths[batch_i]
        hops_mask = batch_hops_mask[batch_i].unsqueeze(0) # [1, max-hop]
        k_pred_relations = batch_pred_relations[batch_i] # [k, max-hop]
        k_relation_accuracy = (k_pred_relations == golden_relation_path).long()
        k_relation_accuracy_mask = k_relation_accuracy * hops_mask 

        k_relation_accuracy_sum = torch.sum(k_relation_accuracy_mask, dim = 1)
        k_equal = (k_relation_accuracy_sum == golden_relation_len).long() 

        if torch.all(k_equal):
            replace_index = random.randint(0, golden_relation_len - 1)
            replace_relation_id  = random.randint(NO_OP_RELATION, len(d_relation2id)- 1)
            neg_relation_path = copy.deepcopy(k_pred_relations[0])
            neg_relation_path[replace_index] = replace_relation_id
            neg_relation_path = neg_relation_path.unsqueeze(0) # [1, max-hop]

            l_negative_samples.append(neg_relation_path)
            l_negative_len.append(golden_relation_len)

        else:
            k_equal_neg = 1 - k_equal
            all_neg_indice = torch.nonzero(k_equal_neg).view(-1)
            neg_relation_path = k_pred_relations[all_neg_indice[0]] 
            equal_self_loop = (neg_relation_path == NO_OP_RELATION).long()
            
            if torch.any(equal_self_loop): 
                all_self_loop_indice = torch.nonzero(equal_self_loop).view(-1)
                negative_len = all_self_loop_indice[0] + 1
            else:
                equal_dummy = (neg_relation_path == 0).long()
                if torch.any(equal_dummy):
                    all_dummy_indice = torch.nonzero(equal_dummy).view(-1)
                    negative_len = all_dummy_indice[0] + 1
                else:
                    negative_len = golden_relation_len 
            
            l_negative_samples.append(neg_relation_path.unsqueeze(0))
            l_negative_len.append(negative_len) 
    
    batch_negative_samples = torch.cat(l_negative_samples, dim = 0)
    batch_negative_len = torch.stack(l_negative_len, dim = 0)
    
    return batch_negative_samples, batch_negative_len