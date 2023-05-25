# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:53:48 2021

@author: Mo
"""

import math
from torch.nn import Linear
from torch_geometric.nn.norm import GraphNorm
# from torch_geometric.nn.models import MLP
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import inspect
from torch.nn import Linear, ReLU, Sequential
from search_algo.predictor_model import compute_metrics
from search_algo.utils import rmse
from settings.config_file import *
import torch.nn as nn


class GNN_Model(torch.nn.Module):
    def __init__(self, param_dict,n_output=1,n_filters=32):
        super(GNN_Model, self).__init__()
        self.aggr1 = param_dict['aggregation1']
        self.aggr2 = param_dict['aggregation2']
        self.hidden_channels = int(param_dict['hidden_channels'])
        self.head1 = param_dict['multi_head1']
        self.head2 = param_dict['multi_head2']
        self.type_task = param_dict['type_task']
        self.gnnConv1 = param_dict['gnnConv1'][0]
        self.gnnConv2 = param_dict['gnnConv2'][0]
        self.global_pooling = param_dict['global_pooling']
        self.edge_attr_size = 0

        
        if self.edge_attr_size == 0:
            self.edge_attr_size = 1

        self.activation1 = param_dict['activation1']
        self.activation2 = param_dict['activation2']
        self.dropout = param_dict['dropout']
        self.normalize1 = param_dict['normalize1']
        self.normalize2 = param_dict['normalize2']

        # Define convolution operation for layer1  and their parameters
        # multiple attention head is used and node and edge features have the same dimensiion
        if param_dict['gnnConv1'][1] == "linear":
            self.conv1 = self.gnnConv1(78, self.hidden_channels)
            self.input_conv2 = self.hidden_channels
        else:
            if 'head' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, head=self.head1, aggr=self.aggr1,
                                               dropout=self.dropout)
                else:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, head=self.head1, aggr=self.aggr1)
                self.input_conv2 = self.hidden_channels * self.head1

            elif 'heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, heads=self.head1, aggr=self.aggr1,
                                               dropout=self.dropout)
                else:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, heads=self.head1, aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels * self.head1

            elif 'num_heads' in inspect.getfullargspec(self.gnnConv1)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, num_heads=self.head1, aggr=self.aggr1,
                                               dropout=self.dropout)
                else:
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, num_heads=self.head1, aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels * self.head1

            else:
                if param_dict['gnnConv1'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, K=2, aggr=self.aggr1)
                elif param_dict['gnnConv1'][1] == 'ChebConv' or param_dict['gnnConv1'][1] == 'gat_sym':
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, K=2, aggr=self.aggr1)
                elif param_dict['gnnConv1'][1] == 'GINConv':
                    nn1= Sequential(Linear(78,  self.hidden_channels), ReLU(),
                              Linear(self.hidden_channels,  self.hidden_channels))
                    self.conv1 = self.gnnConv1(nn1)
                elif param_dict['gnnConv1'][1] == "FastRGCNConv":  # splineconv does not support multihead
                    self.conv1 = self.gnnConv1(78, self.hidden_channels, num_relations=1)

                else:
                    if 'dropout' in inspect.getfullargspec(self.gnnConv1)[0]:
                        self.conv1 = self.gnnConv1(78, self.hidden_channels, aggr=self.aggr1, dropout=self.dropout)
                    else:
                        self.conv1 = self.gnnConv1(78, self.hidden_channels)  # ,aggr=self.aggr1)

                self.input_conv2 = self.hidden_channels

        # multiple attention head is used and node and edge features have the same dimensiion 
        if param_dict['gnnConv2'][1] == "linear":
            self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels)
            self.output_conv2 = self.hidden_channels
        else:

            if 'head' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, head=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, head=self.head2,
                                               aggr=self.aggr2)

                self.output_conv2 = self.hidden_channels * self.head2

            elif 'heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, heads=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, heads=self.head2,
                                               aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels * self.head2


            elif 'num_heads' in inspect.getfullargspec(self.gnnConv2)[0]:
                if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, num_heads=self.head2,
                                               aggr=self.aggr2, dropout=self.dropout)
                else:
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, num_heads=self.head2,
                                               aggr=self.aggr2)
                self.output_conv2 = self.hidden_channels * self.head2

            else:
                if param_dict['gnnConv2'][1] == "SGConv":  # splineconv does not support multihead
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, K=2, aggr=self.aggr2)
                elif param_dict['gnnConv2'][1] == 'ChebConv' or param_dict['gnnConv2'][1] == 'gat_sym':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, K=2, aggr=self.aggr2)
                elif param_dict['gnnConv2'][1] == 'GINConv':
                    nn2 = Sequential(Linear(self.input_conv2, self.hidden_channels), ReLU(),
                                     Linear(self.hidden_channels, self.hidden_channels))
                    self.conv2 = self.gnnConv2(nn2)

                elif param_dict['gnnConv2'][1] == 'FastRGCNConv':
                    self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, num_relations=1)
                else:
                    if 'dropout' in inspect.getfullargspec(self.gnnConv2)[0]:
                        self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, aggr=self.aggr2,
                                                   dropout=self.dropout)
                    else:
                        self.conv2 = self.gnnConv2(self.input_conv2, self.hidden_channels, aggr=self.aggr2)

                self.output_conv2 = self.hidden_channels

        # self.batchnorm1=BatchNorm1d(self.input_conv2)

        self.batchnorm1 = self.normalize1(self.input_conv2)


        self.batchnorm2 = self.normalize2(self.output_conv2)
        self.graphnorm = GraphNorm(self.output_conv2)
        self.linear = Linear(self.output_conv2, 1)


        self.n_output = n_output

        self.mlp_x = Sequential(Linear(self.output_conv2, 1024), ReLU(),
                              Linear(1024, self.output_conv2))

        self.mlp_target = Sequential(Linear(735, 1024), ReLU(),
                              Linear(1024, self.output_conv2))

        self.out1 = Sequential(Linear(self.output_conv2*2, +self.hidden_channels), ReLU(),
                              Linear(+self.hidden_channels, 1))

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        dim0 =n_filters * 4
        self.pool_xt_3 = nn.MaxPool1d(3)
        if config["dataset"]["dataset_name"]=="CCLE":
            self.fc1_xt = nn.Linear(7296, self.output_conv2)
        elif  config["dataset"]["dataset_name"]=="GDSC":
            self.fc1_xt = nn.Linear(7296, self.output_conv2)
        # combined layers
        self.fc1 = nn.Linear(2 * self.output_conv2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out2 = nn.Linear(128, 1)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout,inplace=False)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        # x=self.dropout2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = self.batchnorm2(x)
        x = self.activation2(x)
        # x = self.dropout2(x)


        # 2. Readout layer

        x = self.global_pooling(x, batch)  # [batch_size, self.hidden_channels]
        x = self.mlp_x(x)


        target = data.target
        target = target[:, None, :]
        # 1d conv layers
        conv_xt = self.conv_xt_1(target)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout2(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout2(xc)
        out = self.out2(xc)
        out = nn.Sigmoid()(out)

        return out

def train_function(model, train_loader, criterion, optimizer,epoch=1):
    avg_loss = []
    loss_all = 0
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        inputs = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(inputs)  # Perform a single forward pass.
        y=data.y.view(-1,1)
        train_loss = criterion(out, torch.Tensor.float(y))
        train_loss.backward()
        loss_all += data.num_graphs * train_loss.item()
        optimizer.step()  # Update parameters based on gradients.
        # avg_loss.append(train_loss.item())
    return loss_all / int(config["dataset"]["len_traindata"])

@torch.no_grad()
def test_function(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)

        y_true.append(torch.Tensor.float(data.y.view(-1,1)).cpu().numpy().tolist())
        y_pred.append(pred.cpu().numpy().tolist())
    y_true = np.concatenate(y_true,axis=None)
    y_pred = np.concatenate(y_pred,axis=None)

    performance = compute_metrics(y_true, y_pred)
    return performance