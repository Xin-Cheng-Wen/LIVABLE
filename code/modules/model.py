
import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f
from dgl.nn import GraphConv, AvgPooling, MaxPooling
import dgl
import dgl.function as fn
import math
import numpy as np
#from graph_transformer_edge_layer import GraphTransformerLayer
from mlp_readout import MLPReadout
from torch.autograd import Variable
from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, RelGraphConv
#from dgl.nn.pytorch.conv import APPNPConv
from appnpconv import APPNPConv


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        #pos_enc_dim = 10
        #self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, input_dim)
        '''
        n_layers = 3
        num_head = 10
        self.n_layers = n_layers
        
        self.gtn =  nn.ModuleList([GraphTransformerLayer(input_dim, output_dim, num_heads = num_head,
                                              dropout = 0.2,
                                         max_edge_types = max_edge_types, layer_norm= False,
                                         batch_norm= True, residual= True)
                                   for _ in range (n_layers - 1)])
        '''
        k = 16
        alpha = 0.1
        self.hidden_dim = 128

        self.appnp = APPNPConv(k=k, alpha=alpha, edge_drop=0.5)
        #self.sigmoid = nn.Softmax()
        
        self.hidden_dim2 = 256
        self.batch_size = 64
        self.num_layers = 1
        self.MPL_layer = MLPReadout(self.hidden_dim2, 31)

        self.bigru = nn.GRU(self.inp_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.seq_dim = 128
        self.seq_length = 512
        self.seq_hid = 512
        
        
        self.bigru1 = nn.GRU(self.seq_dim, self.seq_hid, num_layers=self.num_layers, bidirectional=True,
                            batch_first=True)
        self.hidden = self.init_hidden()
        self.hidden1 = self.init_hidden1()
        
        self.MPL_layer1 = MLPReadout(2 * self.seq_hid, 31)
        
        self.dropout = torch.nn.Dropout(0.0)
        self.dropout1 = torch.nn.Dropout(0.0)
        
        #self.weight = nn.Parameter(torch.ones(1))
        
    def init_hidden(self):
        if True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))
    def init_hidden1(self):
        if True:
            if isinstance(self.bigru1, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.seq_hid).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.seq_hid).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.seq_hid)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.seq_hid))
    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num,  self.hidden_dim2))
        return zeros.cuda()
        

    def forward(self, batch, seq, cuda=False):
        graph, features,_ = batch.get_network_inputs(cuda=cuda)
        '''
        seq = torch.LongTensor(seq.numpy())
        seq = seq.cuda()
        seq = self.word_embeddings(seq)
        print(seq.shape)
        seq = seq.cpu().detach().numpy()
        seq = torch.FloatTensor(seq)
        print(seq.dtype)
        '''
        #seq = self.src_emb(seq.int()).cpu().detach().numpy()
        #seq = seq.view(self.batch_size, self.seq_length, -1)
        #print(seq.shape)
        seq, hidden = self.bigru1(seq, self.hidden1)
        seq = torch.transpose(seq, 1, 2)
        seq1 = f.avg_pool1d(seq, seq.size(2)).squeeze(2)
        seq2 = f.max_pool1d(seq, seq.size(2)).squeeze(2)
        #print(seq.shape)
        
        graph = graph.to(torch.device('cuda:0'))
        
        #print(features.shape)
        st = batch.de_batchify_graphs(features)

        # gru
        st, hidden = self.bigru(st, self.hidden)
        features = batch.en_batchify_graphs(st)
        graph = dgl.add_self_loop(graph)
        
        features = self.appnp(graph, features)
        st = batch.de_batchify_graphs(features)

        st = torch.transpose(st, 1, 2)
        #print(st.shape)
        st1 = f.max_pool1d(st, st.size(2)).squeeze(2)
        #print(st1.shape)
        st2 = f.avg_pool1d(st, st.size(2)).squeeze(2)
        # pooling
        #print(st.shape)
        #print(st.shape)
        '''
        st = st.squeeze(2)
        #print(st.shape)
        st = torch.transpose(st, 1, 2)
        st1 = f.max_pool1d(st, st.size(2)).squeeze(2)
        #print(st1.shape)
        st2 = f.avg_pool1d(st, st.size(2)).squeeze(2)
        '''
        #print(st1.shape)
        
        outputs = self.MPL_layer(self.dropout(st1+st2))

        outputs1= self.MPL_layer1(self.dropout1(seq1 + seq2))
        #outputs = nn.Softmax(dim=1)(outputs1+outputs)
        #outputs = nn.Softmax(dim=1)(outputs)
        return outputs1 + outputs, outputs1 + outputs, outputs1 + outputs #+ 0.5 * self.weight * outputs




class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    #前向传播函数
    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result
