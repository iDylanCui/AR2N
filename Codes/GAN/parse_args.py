import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument("--dataset",
                        type=str,
                        # default = 'MetaQA-1H',
                        default = 'MetaQA-2H',
                        # default = 'MetaQA-3H',
                        # default = 'MetaQA-mix',
                        # default = "PQ-2H",
                        # default = "PQ-3H",
                        # default = "PQ-mix",
                        # default = "PQL-2H",
                        # default = "PQL-3H",
                        # default = "PQL-mix",
                        help="dataset for training")

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--num_workers', type=int, default=2, help="Dataloader workers")

argparser.add_argument('--train', action='store_true',
                    help='evaluate the results on the test set')

argparser.add_argument('--eval', action='store_true',
                    help='evaluate the results on the test set')

argparser.add_argument('--add_reversed_edges', action='store_true',
                    help='add reversed edges to extend training set')

# general parameters
argparser.add_argument("--max_hop",
                        type=int,
                        default=3,
                        help="max reasoning hop")

argparser.add_argument("--num_wait_epochs",
                        type=int,
                        default=1,
                        help="valid wait epochs")

argparser.add_argument('--entity_dim', type=int, default=200,
                    help='entity embedding dimension')

argparser.add_argument('--relation_dim', type=int, default=200,
                    help='relation embedding dimension')

argparser.add_argument('--word_dim', type=int, default=300,
                    help='word embedding dimension')

argparser.add_argument('--word_dropout_rate', type=float, default=0.3,
                    help='word embedding dropout rate')

argparser.add_argument('--word_padding_idx', type=int, default=0,
                    help='word padding index')

argparser.add_argument('--is_train_emb', type=bool, default=True,
                    help='train word/entity/relation embedding or not')

argparser.add_argument('--grad_norm', type=float, default=50,
                    help='norm threshold for gradient clipping')

argparser.add_argument('--emb_dropout_rate', type=float, default=0.3,
                    help='Knowledge graph embedding dropout rate')

# Transformer parameters
argparser.add_argument('--head_num', type=int, default=4,
                    help='Transformer head number')

argparser.add_argument('--hidden_dim', type=int, default=100,
                    help='Transformer hidden dimension')

argparser.add_argument('--encoder_layers', type=int, default=2,
                    help='Transformer encoder layers number')

argparser.add_argument('--encoder_dropout_rate', type=float, default=0.3,
                    help='Transformer encoder dropout rate')

# Reinforce Learning
argparser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='weight decay rate')

argparser.add_argument('--history_dim', type=int, default=200,
                    help='path encoder LSTM hidden dimension')

argparser.add_argument('--relation_only', type=bool, default=False,
                    help='search with relation information only, ignoring entity representation')

argparser.add_argument('--rl_dropout_rate', type=float, default=0.3,
                    help='reinforce learning dropout rate')

argparser.add_argument('--history_layers', type=int, default=2,
                    help='path encoder LSTM layers')

argparser.add_argument('--gamma', type=float, default=0.95,
                    help='moving average weight') 

argparser.add_argument('--tau', type=float, default=1.00,
                    help='GAE tau')

# GAN参数
argparser.add_argument('--bucket_interval', type=int, default=5,
                    help='adjacency list bucket size')

argparser.add_argument('--batch_size', type=int, default=512,
                    help='mini-batch size')

# 模型可调节参数
# 固定
argparser.add_argument('--use_coverage_attn', type=bool, default=True, help='use coverage attention or nots')

argparser.add_argument('--use_relation_aware_dis', type=bool, default=True, help='use relation aware question representation in discriminator')

argparser.add_argument('--coverage_coefficient', type=float, default=0.0,
                    help='coverage loss coefficient')

argparser.add_argument('--gen_pretrain_epoch', type=int, default=1,
                    help='generator pretrain epoch number.')

argparser.add_argument('--dis_pretrain_epoch', type=int, default=5,
                    help='discriminator pretrain epoch number.')

argparser.add_argument('--total_epoch', type=int, default=25,
                    help='adversarial learning epoch number.')

argparser.add_argument('--lr_generator_pretrain', type=float, default=0.0005, help='learning rate')

argparser.add_argument('--lr_discriminator_pretrain', type=float, default=0.0005, help='learning rate')

argparser.add_argument('--lr_generator_adversarial', type=float, default=0.0001, help='learning rate')

argparser.add_argument('--lr_discriminator_adversarial', type=float, default=0.0001, help='learning rate')

argparser.add_argument('--beam_size', type=int, default=3, help='size of beam used in beam search inference')
argparser.add_argument('--beam_size_dis', type=int, default=3, help='size of beam used in beam search inference in discriminator')

argparser.add_argument('--use_entity_embedding_vn', type=bool, default=True, help='use entity embedding in value netwok or not')

argparser.add_argument('--use_actor_critic', type=bool, default=True, help='use actor critic optimization.')

argparser.add_argument('--use_gae', type=bool, default=True, help='use gae in actor critic optimization.')

# 可调节
argparser.add_argument('--G_steps', type=int, default=5,
                    help='generator adversarial learning steps.')

argparser.add_argument('--D_steps', type=int, default=5,
                    help='discriminator adversarial learning steps.')

argparser.add_argument('--seed', type=int, default=2021, help='random seed')

argparser.add_argument('--value_loss_coef', type=float, default=0.1,
                    help = "value loss coefficient") 

argparser.add_argument('--G_pretrain_step', type=int, default=1, help='load pertrained generator at the N step')

# 影响模型结构的参数  
argparser.add_argument('--reward_calculation_type', type=int, default=2, help='three typies for calculating rewards')

argparser.add_argument('--reward_coefficient', type=float, default=0.5,help='reward coefficient to balance discriminator and predicted reward')

args = argparser.parse_args()