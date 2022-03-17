# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:39:02 2022

@author: James
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import numpy as np
import time
import random
from sklearn.metrics import f1_score

import os

from utils import load_cora
from models import Encoder,SupervisedGraphSage


if torch.cuda.is_available():
    device = torch.device('cuda')

# device=torch.device('cpu')
seed=1
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)


num_nodes = 2708
feat_data, labels, adj_lists = load_cora()
# features = torch.FloatTensor(feat_data)
features = nn.Embedding(2708, 1433)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
features=features.to(device=device)
labels=Variable(torch.LongTensor(labels)).to(device=device)


# num_nodes,feature_dim=features.shape
num_nodes,feature_dim=features.weight.shape
hidden_dim=256
# hidden_dim=None
embed_dim=128
num_samples = 10
num_classes=7
# agg_method='mean'
agg_method='pool'
graphsage = SupervisedGraphSage(feature_dim,embed_dim,
                                num_classes,adj_lists,agg_method,
                                num_sample=num_samples, hidden_dim=hidden_dim,
                                gcn=True, cuda=True)
graphsage=graphsage.to(device=device)


rand_indices = np.random.permutation(num_nodes)
test = rand_indices[:1000]
val = rand_indices[1000:1500]
train = list(rand_indices[1500:])

optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.5)
times = []
for batch in range(120):
    batch_nodes = train[:256]
    random.shuffle(train)
    start_time = time.time()
    optimizer.zero_grad()
    loss = graphsage.loss(batch_nodes, features,labels[np.array(batch_nodes)])
    loss.backward()
    clip_grad_norm_(filter(lambda p : p.requires_grad, graphsage.parameters()), 1)
    optimizer.step()
    end_time = time.time()
    times.append(end_time-start_time)
    print(batch, loss.item())

val_output = graphsage.forward(val,features) 
print("Validation F1:", f1_score(labels[val].cpu(), val_output.data.cpu().numpy().argmax(axis=1), average="micro"))
print("Average batch time:", np.mean(times))



