import argparse
import logging
import os
import pickle
import sys

os.chdir(sys.path[0])

import numpy as np
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, AdamW

from data_loader.dataset import DataSet
from modules.model import DevignModel
#from trainer_speed import train

#from trainer_sta import train
from trainer_test import train
#from trainer_tsne import train

from utils import tally_param, debug, set_logger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from config import cfg, update_config
from loss import *

import math
from torch.optim.optimizer import Optimizer, required
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.optim



if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='multi')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser', default='../our_word2vec_multi')
    # parser.add_argument('--log_dir', default='devign_FFmpeg.log', type=str)
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=128)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=64)
    parser.add_argument(
        "--cfg",
        type=str,
        help="decide which cfg to use",
        required=False,
        default='configs/bsce.yaml',
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    update_config(cfg, args)   
    
    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    new_model_dir = os.path.join(model_dir, cfg.MODELDIR)
    if not os.path.exists(new_model_dir):
        os.makedirs(new_model_dir)
        
    #设置日志输出
    log_dir = os.path.join(new_model_dir, cfg.LOGNAME)
    if not os.path.exists(log_dir):
        file = open(log_dir, 'w')
        file.flush()
        file.close()
    set_logger(log_dir)
    
    logging.info('Check up feature_size: %d', args.feature_size)
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        logging.info('Warning!!! Graph Embed dimension should be at least equal to the feature dimension')
        args.graph_embed_size = args.feature_size

    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'multi_128_64batch_31_2.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, './multi-train1-v0.json'),
                          valid_src=os.path.join(input_dir, './multi-valid-v0.json'),
                          test_src=os.path.join(input_dir, './multi-test-v0.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)   #../dataset/FFmpeg_input/
        file.close()
    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches), len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)
    '''
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    '''
    logging.info('Check up model_type: ' + args.model_type)
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        #使用DevignModel

        '''
        dataset.feature_size : 100
        args.graph_embed_size : 200
        args.num_steps : 6
        dataset.max_edge_type : 4
        '''


        model = DevignModel(input_dim= 128, output_dim=128,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    #device = torch.device("cuda:2")
    #model.to(device = device)
    model.cuda()

    
    # loss_function = BalancedSoftmaxCE()
    num_classes = 31
    num_class_list = [917, 1140, 469, 383, 205, 275, 249, 134, 164, 110, 138, 109, 80, 95, 75, 49, 32, 34, 31, 26, 26, 24, 20, 18, 12, 16, 10, 12, 25, 13, 133]
    # 8, 16, 8, 6, 3, 12, 11, 4, 7, 2, 5, 5, 6, 2, 1, 3, 3, 3, 1, 4, 3, 2, 2, 1, 1, 3, 2, 0, 1, 2, 2, 0, 1, 1, 0, 0, 1, 1]
    device = torch.device("cuda")
    para_dict = {                                # 我这里只用来调整Loss相关的参数
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }
    loss_function = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    print(loss_function)
    print("loss type:%s"%(cfg.LOSS.LOSS_TYPE))
    #loss_function = CrossEntropyLoss()
    #loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1.2])).float(),reduction='sum')
    
    #loss_function = BCELoss(weight=torch.tensor([1.2]), reduction='sum')
    loss_function.cuda()
    LR = 1e-4
    #optim = Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    #optim = RAdam(model.parameters(),lr=LR,weight_decay=1e-6) 
    #logging.info('Start to train!')
    #开始训练模型
    train(model=model, dataset=dataset, epoches=50, dev_every=len(dataset.train_batches),
          loss_function=loss_function, optimizer=optim,
          save_path=os.path.join(new_model_dir, cfg.MODELNAME), max_patience=20, log_every=5)  #models/FFmpeg/GGNNSumModel....

