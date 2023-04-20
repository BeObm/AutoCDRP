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
from search_algo.predictor_model import evaluate_model_predictor
from search_algo.utils import rmse
from settings.config_file import *
import torch.nn as nn


class GraphRegression(MessagePassing):
    def __init__(self, param_dict,n_output=1,n_filters=32):
        super(GraphRegression, self).__init__()
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

        self.mlp_x = Sequential(Linear(self.output_conv2, 512), ReLU(),
                              Linear(512, self.output_conv2))

        self.mlp_target = Sequential(Linear(735, 512), ReLU(),
                              Linear(512, self.output_conv2))

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
        self.fc1_xt = nn.Linear(2176, self.output_conv2)#2944

        # combined layers
        self.fc1 = nn.Linear(2 * self.output_conv2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out2 = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=self.dropout,inplace=False)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
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


        # # protein input feed-forward
        # target = data.target
        # target =self.mlp_target(target)
        #
        # # Concatenate drug feature with CCL feature
        # out = torch.cat((x, target), dim=1)
        # x = self.dropout2(x)
        # #  Apply a final regression
        # out = self.out1(out)
        # out = nn.Sigmoid()(out)
        #
        # return out
        # protein input feed-forward:
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

def train_gc(model, train_loader, criterion, optimizer,epoch=1):
    avg_loss = []

    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        inputs = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(inputs)  # Perform a single forward pass.
        train_loss = criterion(out, data.y.view(-1, 1).float().to(device))
        if torch.cuda.device_count() > 1 and config["param"]["use_paralell"] == "yes":
            train_loss.mean().backward()
        else:
            train_loss.backward()
        optimizer.step()  # Update parameters based on gradients.
        avg_loss.append(train_loss.item())
        # if batch_idx % log_interval == 0:
        #     print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss.item()))
    return sum(avg_loss) / len(avg_loss)


@torch.no_grad()
def test_gc(model, test_loader, paralell=True):
    model.eval()
    ped_list, label_list = [], []
    for data in test_loader:
        data= data.to(device)
        pred = model(data)
        ped_list = np.append(ped_list, pred.cpu().detach().numpy())
        label_list = np.append(label_list, data.y.cpu().detach().numpy())
    RMSE,pearson,kendalltau,spearmanr = evaluate_model_predictor(label_list, ped_list, "Sampling_distribution")
    return RMSE,pearson,kendalltau,spearmanr


