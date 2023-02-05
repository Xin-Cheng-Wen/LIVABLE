import copy
import json
import logging
import sys
import os
os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from utils import load_default_identifiers, initialize_batch, debug
import numpy
##for each function
class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target, sequence):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.seq = sequence
        print(len(sequence))
        features_new = self.features[:,:128]
        self.graph.add_nodes(self.num_nodes, data={'features': features_new})   ##
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])}) 
            
            #self.graph.add_edge(t, s, data={'etype': torch.LongTensor([etype_number])}) ##
        #print(self.graph)

class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None, g_ident=None, l_ident=None, s_ident = None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident, self.s_ident = load_default_identifiers(n_ident, g_ident, l_ident, s_ident)
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()

    def initialize_dataset(self):

        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src, test_src):
        debug('Reading Train File!')
        #logging.info('train:' + train_src + '; valid:' + valid_src + '; test:' + test_src)
        i = 0
        po_number = 0
        un_number = 0
        class_list = [0] * 68
        with open(train_src,"r") as fp:
            train_data = []
            #for i in fp.readlines():
            #    train_data.append(json.loads(i))
            #for line in fp.readlines(): 
            #    train_data.append(json.loads(line))
            train_data = json.load(fp) 
            #train_data = json.load(open(train_src, 'r'))
            
            for entry in tqdm(train_data):
                if entry[self.l_ident][0][0] <30:
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0], sequence = entry[self.s_ident])
                    print(entry[self.l_ident][0][0])
                    class_list[entry[self.l_ident][0][0]] = class_list[entry[self.l_ident][0][0]] + 1
                else:
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=30, sequence = entry[self.s_ident])
                    print(entry[self.l_ident][0][0])
                    class_list[30] = class_list[30] + 1
                #elif po_number < un_number: 
                #    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                #                        edges=entry[self.g_ident], target= 1)
                #    po_number = po_number + 1
                '''
                else:
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target= 0)
                    un_number = un_number + 1 
                '''             
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
            print(class_list)
            f=open('list.txt','w')
            for n in range(len(class_list)):
                f.write(str(class_list[n])+", ")
            f.close()
        if valid_src is not None:
            debug('Reading Validation File!')
            
            with open(valid_src,"r") as fp:
                valid_data = []
                #for i in fp.readlines():
                #    valid_data.append(json.loads(i))
                #valid_data = json.load(open(valid_src, 'r'))

                valid_data = json.load(fp) 
                for entry in tqdm(valid_data):
                    if entry[self.l_ident][0][0] <30:
                        
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                            edges=entry[self.g_ident], target=entry[self.l_ident][0][0], sequence = entry[self.s_ident])
                    else:
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                            edges=entry[self.g_ident], target=30, sequence = entry[self.s_ident])
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = []
                #for i in fp.readlines():
                #    test_data.append(json.loads(i))
                test_data = json.load(fp) 
                #test_data = json.load(open(test_src, 'r'))
                for entry in tqdm(test_data):
                    if entry[self.l_ident][0][0] <30:
                        
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                            edges=entry[self.g_ident], target=entry[self.l_ident][0][0], sequence = entry[self.s_ident])
                    else:
                        example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                            edges=entry[self.g_ident], target=30, sequence = entry[self.s_ident])
                    self.test_examples.append(example)


    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size

        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)


        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()

        i = 0
        for entry in taken_entries:
            seq = torch.FloatTensor(entry.seq).view(-1, 128)
            #print(seq.shape)
            if len(entry.seq) < 128 * 512:
               k = torch.zeros(512-seq.shape[0], 128)
               seq = torch.cat((seq,k),dim = 0)
            if len(entry.seq) > 128 * 512:
               seq = seq[:512,:]
            if i == 0:
                seq1 = seq.view(1, -1,128)
                i = i + 1
            else: seq1 = torch.cat((seq1, seq.view(1, -1,128)), dim = 0)
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels), seq1

    def get_next_train_batch(self):

        #print(len(self.train_batches))
        if len(self.train_batches) == 0:
            #print('k'*40)
            self.initialize_train_batch()

        ids = self.train_batches.pop()
        if(len(self.train_batches) == 1):
            ids1 = self.train_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        if (len(self.valid_batches) == 1):
            ids1 = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        if (len(self.test_batches) == 1):
            ids1 = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
