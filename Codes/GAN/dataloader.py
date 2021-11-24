import torch
from torch.utils.data import Dataset, DataLoader

class Dataset_GAN(Dataset):
    def __init__(self, data, entity_num, isTrain, is_Pretrain_gen):
        self.data = data # list
        self.vec_len = entity_num
        self.isTrain = isTrain
        self.is_Pretrain_gen = is_Pretrain_gen

    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        one_hot = torch.LongTensor(self.vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        topic_entity_id = data_point[0]
        question_pattern_id = data_point[1]
        answer_entities_id = data_point[2]
        relations_id = data_point[3]
        answer_entities_onehot = self.toOneHot(answer_entities_id)

        if self.isTrain and self.is_Pretrain_gen:
            intermediate_entities_id = data_point[4]
            
            return torch.LongTensor(question_pattern_id), topic_entity_id, answer_entities_onehot, torch.LongTensor(relations_id), torch.LongTensor(intermediate_entities_id)
        else:
            return torch.LongTensor(question_pattern_id), topic_entity_id, answer_entities_onehot, torch.LongTensor(relations_id)
      

def _collate_fn(batch):
    NO_OP_RELATION = 2 # self-loop relation id
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)
    is_train_Pretrain_gen = True if len(sorted_seq[0]) == 5 else False
    
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]
    longest_sample = sorted_seq_lengths[0]
    longest_relation_path = max([len(i[3]) for i in sorted_seq])

    minibatch_size = len(batch) 
    input_lengths = []
    relation_path_lens = []
    p_head = []
    p_tail = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)
    relations = torch.zeros(minibatch_size, longest_relation_path, dtype=torch.long) + NO_OP_RELATION
    intermediate_entities = torch.zeros(minibatch_size, longest_relation_path, dtype=torch.long) 
    for x in range(minibatch_size):
        sample = sorted_seq[x][0]
        seq_len = len(sample)
        input_lengths.append(seq_len)
        inputs[x].narrow(0,0,seq_len).copy_(sample)

        p_head.append(sorted_seq[x][1])

        tail_onehot = sorted_seq[x][2]
        p_tail.append(tail_onehot)

        relations_id = sorted_seq[x][3]
        relation_len = len(relations_id)
        relation_path_lens.append(relation_len)
        relations[x].narrow(0,0,relation_len).copy_(relations_id)

        if is_train_Pretrain_gen:
            intermediate_entities_id = sorted_seq[x][4]
            intermediate_entities[x].narrow(0,0,relation_len).copy_(intermediate_entities_id)
            intermediate_entities[x][relation_len:] = intermediate_entities[x][relation_len-1]


    if is_train_Pretrain_gen:
        return inputs, torch.LongTensor(input_lengths), torch.LongTensor(p_head), torch.stack(p_tail), relations, torch.LongTensor(relation_path_lens), intermediate_entities # inputs: [batch, max_tokens]; input_lengths: [batch]; p_head: [batch]; p_tail: [batch, entities_num]; relations = intermediate_entities: [batch, max_relation_path]
    else:
        return inputs, torch.LongTensor(input_lengths), torch.LongTensor(p_head), torch.stack(p_tail), relations, torch.LongTensor(relation_path_lens)


class DataLoader_GAN(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_GAN, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn