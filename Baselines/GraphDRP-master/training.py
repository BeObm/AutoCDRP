import math

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.mgataf import MGATAFNet
from models.gat_gcn_transformer import GAT_GCN_Transformer
from models.supergat import SuperGATNet
from models.chebnet import ChebNet
from models.arma import ARMA
from utils import *
from torch.optim.lr_scheduler import  ReduceLROnPlateau
import datetime
import argparse


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def load_dataset(train_batch, val_batch, test_batch, lr, num_epoch, dataset, experiment):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    processed_data_file_train = 'data/' + dataset + '/processed/' + f'{dataset}_{experiment}_train.pt'
    processed_data_file_val = 'data/' + dataset + '/processed/' + f'{dataset}_{experiment}_val.pt'
    processed_data_file_test = 'data/' + dataset + '/processed/' + f'{dataset}_{experiment}_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (
            not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
        exit()
    else:
        train_data = TestbedDataset(root='data/' + dataset, dataset=f'{dataset}_{experiment}_train')
        val_data = TestbedDataset(root='data/' + dataset, dataset=f'{dataset}_{experiment}_val')
        test_data = TestbedDataset(root='data/' + dataset, dataset=f'{dataset}_{experiment}_test')

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
        print("CPU/GPU: ", torch.cuda.is_available())
    return train_loader, val_loader, test_loader


def main(modeling, train_loader, val_loader, test_loader, lr, num_epoch, log_interval, cuda_name, dataset, experiment):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    model_st = modeling.__name__
    os.makedirs(f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}", exist_ok=True)

    train_losses = []
    val_losses = []
    val_pearsons = []
    print('\nrunning on ', model_st + '_' + dataset)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(device)
    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_mse = 1000
    best_pearson = 1
    best_epoch = -1
    model_file_name = f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/model.model"
    result_file_name = f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/result.csv"
    loss_fig_name = f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/loss"
    pearson_fig_name = f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/pearson"
    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

        G, P = predicting(model, device, val_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

        G_test, P_test = predicting(model, device, test_loader)
        ret_test = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test)]

        train_losses.append(train_loss)
        val_losses.append(ret[1])
        val_pearsons.append(ret[2])
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret_test)))
            best_epoch = epoch + 1
            best_mse = ret[1]
            best_pearson = ret[2]
            print(' rmse improved at epoch ', best_epoch, '; best_rmse:', math.sqrt(best_mse), model_st, dataset)
        else:
            print(' no improvement since epoch ', best_epoch, '; best_rmse, best pearson:', math.sqrt(best_mse), best_pearson,
                  model_st, dataset)
        scheduler.step(ret[2])

    draw_loss(train_losses, val_losses, loss_fig_name)
    draw_pearson(val_pearsons, pearson_fig_name)
    dfLoss = pd.DataFrame(list(zip(train_losses, val_losses)), columns=["train_loss", "val_loss"])
    dfLoss.to_csv(f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/train_loss_log.csv")
    dfpcc = pd.DataFrame(list(zip(val_pearsons, range(len(val_pearsons)))), columns=["train_loss", "val_loss"])
    dfpcc.to_csv(f"Baselines/GraphDRP-master/Baseline_Logs/{dataset}/{experiment}/{model_st}/train_pcc_log.csv")
    with open("results_log.txt", "a") as rs:
        rs.write(
            f"{'*' * 10}  Experiment result on {dataset} dataset| {experiment} experiment with {model_st}  {'*' * 10} \n")
        rs.write(f" Best PCC = {best_pearson}")
        rs.write(f" Best RMSE = {math.sqrt(best_mse)}")
        rs.write(f" Best epoch = {best_epoch} \n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=8,
                        help='0: GINConvNet, 1: GATNet, 2: GAT_GCN, 3: GCNNet')
    parser.add_argument('--train_batch', type=int, required=False, default=512, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=512, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=512, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    parser.add_argument('--dataset', type=str, required=False, default="CCLE", help='dataset name')
    parser.add_argument('--experiment', type=str, required=False, default="mix", help='type of experiment',
                        choices=["mix", "drug_blind", "cell_blind"])

    args = parser.parse_args()
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    train_loader, val_loader, test_loader = load_dataset(train_batch, val_batch, test_batch, lr, num_epoch,
                                                         args.dataset, args.experiment)
    modelings = [ GCNNet, GINConvNet, GATNet, GAT_GCN,SuperGATNet,MGATAFNet, GAT_GCN_Transformer,ARMA,ChebNet]

    for modeling in [modelings[args.model]]:
        main(modeling, train_loader, val_loader, test_loader, lr, num_epoch, log_interval, cuda_name, args.dataset,
             args.experiment)
        torch.cuda.empty_cache()
