# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:39:02 2022

@author: James
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs,features, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)

        if not num_sample is None:
            #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
            #所以在这里最好不要用循环，如果这里用了下面的set.union也得用循环
            # samp_neighs=[]
            # for to_neigh in to_neighs:
            #     if len(to_neigh) >= num_sample:
            #         samp_neighs.append([set(random.sample(to_neigh, num_sample,))])
            #     else:
            #         samp_neighs.append(to_neigh)
            samp_neighs = [set(random.sample(to_neigh, num_sample,)) 
                            if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
        unique_nodes_list = list(set.union(*samp_neighs))
        #n是unique_nodes，i是索引。重新建立的是所有涉及到的node的值和索引的dict
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        #mask shape is B_S(S指的是所有此次batch采样涉及到的节点)
        #每一行是一个需采样节点的所有采出来的neighbor的mask
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #samp_neigh里面每一个节点在unique_nodes里面的索引
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        #row_indices是对所采样的node做了重新编号
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        #mean aggerator取平均值
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            # embed_matrix = features[torch.LongTensor(unique_nodes_list).cuda()]
            embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            #索引采样node的feature
            # embed_matrix = features[torch.LongTensor(unique_nodes_list)]
            embed_matrix = features(torch.LongTensor(unique_nodes_list))
        #mask有些类似子图的adjacency matrix,只不过形状是B_S，每一行是一个需采样节点的所有采出来的neighbor
        #mask shape is B_S, embed_matrxi shape is S_F
        to_feats = mask.mm(embed_matrix)
        return to_feats


class PoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using pooling of neighbors' embeddings
    """
    def __init__(self,input_dim, hidden_dim,pool_fn,gcn=False, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """

        super(PoolAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.pool_fn=pool_fn
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        
    def forward(self, nodes, to_neighs,features, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)

        if not num_sample is None:
            #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
            #所以在这里最好不要用循环，如果这里用了下面的set.union也得用循环
            # samp_neighs=[]
            # for to_neigh in to_neighs:
            #     if len(to_neigh) >= num_sample:
            #         samp_neighs.append([set(random.sample(to_neigh, num_sample,))])
            #     else:
            #         samp_neighs.append(to_neigh)
            samp_neighs = [set(random.sample(to_neigh, num_sample,)) 
                            if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
        unique_nodes_list = list(set.union(*samp_neighs))
        #n是unique_nodes，i是索引。重新建立的是所有涉及到的node的值和索引的dict
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #samp_neigh里面每一个节点的索引
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        #每一个samp_neighs的索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        if self.cuda:
            # embed_matrix = features[torch.LongTensor(unique_nodes_list).cuda()]
            embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())

        else:
            #索引采样node的feature
            # embed_matrix = features[torch.LongTensor(unique_nodes_list)]
            embed_matrix = features(torch.LongTensor(unique_nodes_list))
            
        embed_matrix=self.mlp(embed_matrix)
        #mask有些类似子图的adjacency matrix
        to_feats = self.pool_fn(mask,embed_matrix,self.cuda)
        return to_feats


def pool_max(mask,embed_matrix,use_cuda):
    """
    max pool
    mask shape is B_S
    embed_matrix shape S_F
    """
    assert mask.shape[1]==embed_matrix.shape[0], 'mask shape 1 must match embed_matrix shape 0'
    b,s=mask.shape
    mask=mask.t()
    s,f=embed_matrix.shape
    feats=[]
    for i in range(b):
        temp=mask[:,i].unsqueeze(1).expand(-1,f)
        temp1=(embed_matrix*temp).max(dim=0).values
        feats.append(temp1)
    feats=torch.stack(feats,dim=0)
    return feats


def pool_mean(mask,embed_matrix,use_cuda):
    """
    mean pool
    mask shape is B_S
    embed_matrix shape S_F
    """
    assert mask.shape[1]==embed_matrix.shape[0], 'mask shape 1 must match embed_matrix shape 0'
    b,s=mask.shape
    mask=mask.t()
    s,f=embed_matrix.shape
    feats=[]
    for i in range(b):
        temp=mask[:,i].unsqueeze(1).expand(-1,f)
        temp1=torch.stack(embed_matrix[temp==1].split(f,dim=0),dim=0).mean(dim=0)
        feats.append(temp1)
    feats=torch.stack(feats,dim=0)
    return feats


class LSTMAggregator(nn.Module):
    """
    Aggregates a node's embeddings using LSTM of neighbors' embeddings
    """
    def __init__(self,input_dim, hidden_dim,bidirectional=True,gcn=False, cuda=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        """

        super(LSTMAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), 
                            bidirectional=bidirectional, batch_first=True)
        # self.activation = nn.ReLU()
        
        
    def forward(self, nodes, to_neighs,features, num_sample=10):
        """
        以所有的neighbor为sequence进行LSTM
        """
        # Local pointers to functions (speed hack)

        if not num_sample is None:
            #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
            #所以在这里最好不要用循环，如果这里用了下面的set.union也得用循环
            # samp_neighs=[]
            # for to_neigh in to_neighs:
            #     if len(to_neigh) >= num_sample:
            #         samp_neighs.append([set(random.sample(to_neigh, num_sample,))])
            #     else:
            #         samp_neighs.append(to_neigh)
            samp_neighs = [set(random.sample(to_neigh, num_sample,)) 
                            if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
        unique_nodes_list = list(set.union(*samp_neighs))
        #n是unique_nodes，i是索引。重新建立的是所有涉及到的node的值和索引的dict
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #samp_neigh里面每一个节点的索引
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        #每一个samp_neighs的索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        if self.cuda:
            # embed_matrix = features[torch.LongTensor(unique_nodes_list).cuda()]
            embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())

        else:
            #索引采样node的feature
            # embed_matrix = features[torch.LongTensor(unique_nodes_list)]
            embed_matrix = features(torch.LongTensor(unique_nodes_list))
        
        #mask有些类似子图的adjacency matrix
        #neib_features shape is B_S_F
        agg_neib=self.neib_lstm(mask,embed_matrix)
        return agg_neib


    def neib_lstm(self,mask,embed_matrix):
        """
        mean pool
        mask shape is B_S
        embed_matrix shape S_F
        """
        assert mask.shape[1]==embed_matrix.shape[0], 'mask shape 1 must match embed_matrix shape 0'
        b,s=mask.shape
        mask=mask.t()
        s,f=embed_matrix.shape
        feats=[]
        
        for i in range(b):
            temp=mask[:,i].unsqueeze(1).expand(-1,f)
            #S_F
            temp1=torch.stack(embed_matrix[temp==1].split(f,dim=0),dim=0)
            b=torch.randperm(temp1.size(0))
            temp1=temp1[b]
            temp2,_=self.lstm(temp1.unsqueeze(0))
            # !! Taking final state, but could do something better (eg attention)
            #B_S_F->F
            feats.append(temp2[:,-1,:].squeeze(0).squeeze(0))
        #B_F
        to_feats=torch.stack(feats,dim=0)
        # if self.activation:
        #     to_feats = self.activation(to_feats)
        return to_feats


# 以attention score为权重加权平均所有的sampled neighbors
class AttentionAggregator(nn.Module):
    """
    Aggregates a node's embeddings using attention of neighbors' embeddings
    """
    def __init__(self,input_dim, hidden_dim,gcn=False, cuda=False): 
        """
        以attention score为权重加权平均所有的sampled neighbors
        
        """

        super(AttentionAggregator, self).__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.att1 = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.att2 = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.att3 = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        # self.activation = nn.ReLU()
        
        
    def forward(self, nodes, to_neighs,features, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)

        if not num_sample is None:
            #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
            #所以在这里最好不要用循环，如果这里用了下面的set.union也得用循环
            # samp_neighs=[]
            # for to_neigh in to_neighs:
            #     if len(to_neigh) >= num_sample:
            #         samp_neighs.append([set(random.sample(to_neigh, num_sample,))])
            #     else:
            #         samp_neighs.append(to_neigh)
            samp_neighs = [set(random.sample(to_neigh, num_sample,)) 
                            if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        #set.union求并集时所迭代的list不能是append过的，否则会报unhashable type: 'set'
        unique_nodes_list = list(set.union(*samp_neighs))
        #n是unique_nodes，i是索引。重新建立的是所有涉及到的node的值和索引的dict
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #samp_neigh里面每一个节点的索引
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        #每一个samp_neighs的索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        if self.cuda:
            # embed_matrix = features[torch.LongTensor(unique_nodes_list).cuda()]
            embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())

        else:
            #索引采样node的feature
            # embed_matrix = features[torch.LongTensor(unique_nodes_list)]
            embed_matrix = features(torch.LongTensor(unique_nodes_list))
        
        #mask有些类似子图的adjacency matrix
        #neib_features shape is B_F
        agg_neib=self.neib_attn(mask,embed_matrix)
        return agg_neib


    def neib_attn(self,mask,embed_matrix):
        """
        mask shape is B_S
        embed_matrix shape S_F
        """
        assert mask.shape[1]==embed_matrix.shape[0], 'mask shape 1 must match embed_matrix shape 0'
        b,s=mask.shape
        mask=mask.t()
        s,f=embed_matrix.shape
        feats=[]
        
        for i in range(b):
            temp=mask[:,i].unsqueeze(1).expand(-1,f)
            #S_F
            temp1=torch.stack(embed_matrix[temp==1].split(f,dim=0),dim=0)
            temp2=self.att1(temp1)
            temp3=self.att2(temp1)
            temp4=self.att3(temp1)
            neib_att=F.softmax(temp2@temp3.t(),dim=-1)
            #S_F->F
            temp5=torch.sum(neib_att@temp4,dim=0)
            feats.append(temp5)
        #B_F
        feats=torch.stack(feats,dim=0)
        return feats

