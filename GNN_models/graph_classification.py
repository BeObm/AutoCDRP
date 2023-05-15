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
    def __init__(self, param_dict, n_output=2, n_filters=32):
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


        self.activation1 = param_dict['activation1']
        self.activation2 = param_dict['activation2']
        self.dropout = param_dict['dropout']
        self.normalize1 = param_dict['normalize1']
        self.normalize2 = param_dict['normalize2']

        # Define convolution operation for layer1  and their parameters
        # multiple attention head is used and node and edge features have the same dimensiion


        self.batchnorm1 = self.normalize1(self.hidden_channels)

        self.batchnorm2 = self.normalize2(self.hidden_channels)

        self.n_output = n_output

        self.mlp_x = Sequential(Linear(self.hidden_channels, 512), self.activation1,
                                Linear(512, self.hidden_channels))

        self.mlp_target = Sequential(Linear(735, 512), self.activation1,
                                     Linear(512, self.hidden_channels))

        self.out1 = Sequential(Linear(self.hidden_channels * 2, +self.hidden_channels), self.activation2,
                               Linear(+self.hidden_channels, 1))

        if param_dict['gnnConv1'][1] == "linear":
            self.conv1 = self.gnnConv1(78, self.hidden_channels)
            self.hidden_channels = self.hidden_channels
        else:
            self.conv1 = self.gnnConv1(78, self.hidden_channels, aggr=self.aggr1)
        if param_dict['gnnConv2'][1] == "linear":
            self.conv2 = self.gnnConv2(self.hidden_channels, self.hidden_channels)
            self.hidden_channels = self.hidden_channels
        else:
            self.conv2 = self.gnnConv2(self.hidden_channels, self.hidden_channels, aggr=self.aggr2)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)

        self.pool_xt_3 = nn.MaxPool1d(3)
        if config["dataset"]["dataset_name"] == "CCLE":
            self.fc1_xt = nn.Linear(7296, self.hidden_channels)
        elif config["dataset"]["dataset_name"] == "GDSC":
            self.fc1_xt = nn.Linear(7296, self.hidden_channels)
        # combined layers
        self.fc1 = nn.Linear(2 * self.hidden_channels, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out2 = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout, inplace=False)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 2. Readout layer

        x = self.global_pooling(x, batch)
        x = self.mlp_x(x)

        target = data.target
        target = target[:, None, :]
        # 1d conv layers
        conv_xt = self.conv_xt_1(target)
        conv_xt = self.activation1(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.activation1(conv_xt)
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
        # out = nn.Sigmoid()(out)
        # out = F.log_softmax(out, dim=1)

        return out


def train_function(model, train_loader, criterion, optimizer, epoch=1):
    avg_loss = []
    loss_all = 0
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass.
        data.y = data.y.type(torch.LongTensor).to(device)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()  # Update parameters based on gradients.
        # avg_loss.append(train_loss.item())

    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test_function(model, test_loader, paralell=True):
    model.eval()
    y_true, y_pred = [], []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            print([a for a in out])
            pred = out.argmax(dim=-1)
        data.y = data.y.type(torch.LongTensor).to(device)
        y_true.append(data.y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true, axis=None)
    y_pred = np.concatenate(y_pred, axis=None)

    performance = compute_metrics(y_true, y_pred)
    return performance