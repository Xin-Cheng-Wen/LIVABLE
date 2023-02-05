import torch
from dgl import DGLGraph
import numpy
import dgl
import dgl.function as fn
from torch import nn


class BatchGraph:
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0
        self.graphid_to_nodeids1 = {}

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        #新的结点数量
        num_new_nodes = _g.number_of_nodes()
        #图的id
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))
        self.graphid_to_nodeids1[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))
        #添加新的结点
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        #得到新的边
        sources, dests = _g.all_edges()
        #添加新的源节点
        sources += self.number_of_nodes
        #添加新的目的结点
        dests += self.number_of_nodes
        #添加新的边
        self.graph.add_edges(sources, dests, data=_g.edata)
        #_g  = dgl.add_self_loop(_g)
        #self.graph = dgl.batch([self.graph, _g])
        #添加新的结点数量
        self.number_of_nodes += num_new_nodes
        #添加子图数量
        self.num_of_subgraphs += 1
        #if (self.num_of_subgraphs == 64):
            #print(self.num_of_subgraphs)
            #self.graph  = dgl.add_self_loop(self.graph)
            
            



    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)



    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)
        #print(features)
        #print(self.graphid_to_nodeids.keys())
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        #for i in self.graphid_to_nodeids.keys():
        #    vectors = features.index_select(dim=0, index=self.graphid_to_nodeids[gid])

        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)
        
        return output_vectors#, lengths
        
    def en_batchify_graphs(self, features=None ,):
        assert isinstance(features, torch.Tensor)
        #print(features)
        #print(self.graphid_to_nodeids.keys())
        #vectors = [index=self.graphid_to_nodeids[gid] for gid in
        #           self.graphid_to_nodeids.keys()]
        
        for i in self.graphid_to_nodeids.keys():
           length = len(self.graphid_to_nodeids[i])
           vectors = features[i,0:length,:]
           if i == 0:
               output_vectors = vectors
           else:
               output_vectors = torch.cat((output_vectors, vectors) ,dim=0)
        #   vectors = features.index_select(dim=0, index=self.graphid_to_nodeids[gid])
        #print(output_vectors.shape)
        return output_vectors#, lengths
        
    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')

from scipy import sparse as sp



class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()
        #self.pos_enc_dim = 10
        #self.embedding_lap_pos_enc = nn.Linear(self.pos_enc_dim, 100).to(torch.device('cuda:0'))
    def get_network_inputs(self, cuda=False, device=None):
        
        features = self.graph.ndata['features']
        #图结构信息
        edge_types = self.graph.edata['etype']
        #print(edge_types)
        if cuda:
            #self.cuda(device=device)
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)#, h_lap_pos_enc.cuda(device=device)
        else:
            return self.graph, features, edge_types#, h_lap_pos_enc
        pass
