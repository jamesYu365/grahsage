# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:39:02 2022

@author: James
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from layers import MeanAggregator,PoolAggregator,pool_max,pool_mean


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, feature_dim,
            embed_dim, adj_lists, aggregator,
            num_sample=10, hidden_dim=None,
            gcn=False, cuda=False): 
        super(Encoder, self).__init__()

        self.feat_dim = feature_dim
        self.hidden_dim=hidden_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        
        if self.aggregator.__class__.__name__=='MeanAggregator':
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        elif self.aggregator.__class__.__name__=='PoolAggregator':
            assert hidden_dim!=None,'you must input hidden_dim when using pool aggerator'
            self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.hidden_dim if self.gcn else (self.hidden_dim +self.feat_dim)))

        init.xavier_uniform_(self.weight)

    def forward(self, nodes,features):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              features,self.num_sample)
        if not self.gcn:
            if self.cuda:
                # self_feats = features(torch.LongTensor(nodes).cuda())
                self_feats = features(torch.LongTensor(nodes).cuda())
            else:
                # self_feats = features(torch.LongTensor(nodes))
                self_feats = features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        
        #combined shape is F_B
        combined = F.relu(self.weight.mm(combined.t()))
        return combined.t()



class SupervisedGraphSage(nn.Module):
    def __init__(self, feature_dim,embed_dim,
            num_classes,adj_lists,agg_method,
            num_sample=10, hidden_dim=None,
            gcn=False, cuda=False):
        """
        Simple supervised GraphSAGE model as well as examples running the model
        on the Cora and Pubmed datasets.
        #两层graphsage-pool

        """
        
        super(SupervisedGraphSage, self).__init__()

        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        init.xavier_uniform_(self.weight)
        
        if agg_method=='mean':
            self.agg1 = MeanAggregator(cuda=cuda)
        elif agg_method=='pool':
            self.agg1 = PoolAggregator(feature_dim,hidden_dim,pool_fn=pool_mean, cuda=cuda)
        self.enc1 = Encoder(feature_dim, embed_dim, adj_lists, self.agg1,
                       hidden_dim=hidden_dim, gcn=gcn, cuda=cuda)
       
        if agg_method=='mean':
            self.agg2 = MeanAggregator(cuda=cuda)
        elif agg_method=='pool':
            self.agg2 = PoolAggregator(embed_dim,hidden_dim,pool_fn=pool_mean, cuda=cuda)

        self.enc2 = Encoder(embed_dim, embed_dim, adj_lists, self.agg2,
                       hidden_dim=hidden_dim, gcn=gcn, cuda=cuda)


    def forward(self, nodes,features):
        #采样是从里向外的，聚合是从外向里的
        #embeds2 shape is B_F
        embeds2 = self.enc2(nodes,lambda nodes : self.enc1(nodes,features))
        
        scores = self.weight.mm(embeds2.t())
        return scores.t()

    def loss(self, nodes,features, labels):
        scores = self.forward(nodes,features)
        return self.xent(scores, labels.squeeze())
