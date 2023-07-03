# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:44:23 2021
@author: Mo
"""


from sklearn.metrics import mean_squared_error

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, auc, recall_score, balanced_accuracy_score, accuracy_score, f1_score
from scipy.stats import kendalltau, spearmanr, pearsonr
import math
from scipy import stats
from search_algo.utils import *
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from search_space_manager.sample_models import *
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.nn import global_add_pool  # global_mean_pool, global_max_pool,
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import glob
from sklearn.linear_model import SGDRegressor, LassoCV
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, GENConv

from torch_geometric.nn.norm import GraphNorm
from search_space_manager.map_functions import LinearConv
from settings.config_file import *

set_seed()


def get_prediction(performance_records, e_search_space, regressor_model_type):
    if (config["param"]["predictor_dataset_type"]) == "graph":
        TopK_final = get_prediction_from_graph(performance_records, e_search_space, regressor_model_type)
    elif (config["param"]["predictor_dataset_type"]) == "table":
        TopK_final = get_prediction_from_table(performance_records, e_search_space)
    return TopK_final


class Predictor(MessagePassing):
    def __init__(self, in_channels, dim, out_channels, drop_out):
        super(Predictor, self).__init__()
        #         self.embed_edges = Linear(self.edge_attr_size, self.hidden_channels)
        # print("in channels dim =",in_channels)
        self.conv1 = GATConv(in_channels, dim*3, aggr="mean",heads=5)
        self.conv2 = GATConv(dim*3*5, dim, aggr="mean")
        self.drop_out = drop_out
        # self.normalize = InstanceNorm(dim)
        self.graphnorm = GraphNorm(dim)
        self.linear = Linear(dim, 64)
        self.linear2 = Linear(64, out_channels)

    def forward(self, data):
        x, edge_index, batch= data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)

        x = global_add_pool(x, batch)
        x = self.graphnorm(x)

        x = F.relu(self.linear(x))
        x = self.linear2(x)

        return x


class Predictor6(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, drop_out, conv="GAT"):
        super(Predictor, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.drop_out = drop_out
        self.out_channel = out_channels
        self.conv1, self.conv2,self.conv3 = self.get_conv(conv)
        # self.normalize = InstanceNorm(dim)
        self.graphnorm = GraphNorm(dim)
        self.linear = Linear(dim, 1024)
        self.linear2 = Linear(1024, out_channels)
        self.relu = nn.ReLU()



    def get_conv(self, conv):
        if conv == "GIN":
            nn1 = Sequential(Linear(self.in_channels, self.dim), ReLU(),
                             Linear(self.dim, self.dim))
            nn2 = Sequential(Linear(self.dim, self.dim), ReLU(),
                             Linear(self.dim, self.dim))
            return GINConv(nn1), GINConv(nn2)
        elif conv == "GCN":
            return GCNConv(self.in_channels, self.dim*2), GCNConv(self.dim*2, self.dim*4),GCNConv(self.dim*4, self.dim)
        elif conv == "GAT":
            return GATConv(self.in_channels, self.dim*2, heads=5, dropout=self.drop_out), GATConv(self.dim * 10, self.dim*3,
                                                                                                dropout=self.drop_out,heads=5),GATConv(self.dim *15, self.dim,
                                                                                                dropout=self.drop_out)
        elif conv == "GEN":
            return GENConv(self.in_channels, self.dim*2), GENConv(self.dim*2, self.dim*4),GENConv(self.dim*4, self.dim)
        elif conv == "MLP":
            return LinearConv(self.in_channels, self.dim*4), LinearConv(self.dim*4, self.dim*2),LinearConv(self.dim*2, self.dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # x= self.graphnorm(x)
        x = F.dropout(x, p=self.drop_out, training=True)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # x = self.graphnorm(x)
        x = F.dropout(x, p=self.drop_out, training=True)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        # x = self.graphnorm(x)
        # x = F.dropout(x, p=self.drop_out, training=True)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.drop_out, training=True)
        x = self.linear(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = nn.Sigmoid()(x)
        return x


def trainpredictor(predictor_model, train_loader,optimizer,epoch):
    predictor_model.train()
    avg_loss = []
    # loss_fct=torch.nn.MSELoss(reduction='sum')
    loss_fct = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = predictor_model(data)
        loss = loss_fct(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    loss = loss_all / len(train_loader.dataset)
    return loss


@torch.no_grad()
def testpredictor(model, test_loader, title):
    model.eval()
    y_true = []
    y_pred = []
    for data in test_loader:
        data= data.to(device)
        with torch.no_grad():
            pred = model(data)
        y_true.append(torch.Tensor.float(data.y.view(-1, 1)).cpu().numpy().tolist())
        y_pred.append(pred.cpu().numpy().tolist())
    y_true = np.concatenate(y_true, axis=None)
    y_pred = np.concatenate(y_pred, axis=None)

    predictor_R2_Score, predictor_mse, predictor_mae, predictor_corr = evaluate_model_predictor(y_true,
                                                                                                y_pred,
                                                                                                title)
    return predictor_R2_Score, predictor_mse, predictor_mae, predictor_corr


@torch.no_grad()
def predict_accuracy_using_graph(model, graphLoader):
    model.eval()
    search_metric = config["param"]["search_metric"]
    prediction_dict = {}
    prediction_dict['model_config'] = []
    prediction_dict[search_metric] = []
    k = int(config["param"]["k"])
    i = 0

    for data in graphLoader:
        accuracy = []
        i += 1
        data = data.to(device)
        pred = model(data)
        accuracy = np.append(accuracy, pred.cpu().detach().numpy())
        choices = deepcopy(data.model_config_choices)
        choice = []
        for a in range(len(pred)):  # loop for retriving the GNN configuration of each graph in the data loader
            temp_list = []
            for key, values in choices.items():
                # print(f"this is value {values}")
                # print(f"this is value[1] {values[a]}")
                # print(f"this is value[1][a] {values[a][1]}")
                temp_list.append((key, values[a][1]))
            choice.append(temp_list)
        prediction_dict['model_config'].extend(choice)
        prediction_dict[search_metric].extend(accuracy)
        # print("choice is",choice)
        # print("acc=",round(float(model_acc[1].item()),2),"Acctype=",type(model_acc[1].item()))

    df = pd.DataFrame.from_dict(prediction_dict)
    TopK = df.nsmallest(n=k, columns=search_metric, keep="all")
    return TopK


def get_prediction_from_graph(performance_record, e_search_space, regressor_model_type):
    set_seed()
    dim = int(config["predictor"]["dim"])
    drop_out = float(config["predictor"]["drop_out"])
    lr = float(config["predictor"]["lr"])
    wd = float(config["predictor"]["wd"])
    num_epoch = int(config["predictor"]["num_epoch"])
    k = int(config["param"]["k"])
    predict_sample = int(config["param"]["predict_sample"])
    n_sample = int(config["param"]["n"])
    graphlists = []
    search_metric = config["param"]["search_metric"]
    bestY = 0

    for filename in glob.glob(performance_record + '/*'):
        data = torch.load(filename)
        data.y = data.y.view(-1, 1)

        graphlists.append(data)
        if data.y.item() > bestY:
            bestY = data.y.item()

    print("best Y=", bestY)

    graph_list = deepcopy(graphlists[:n_sample])
    val_size = int(len(graph_list) * 20 / 100)

    val_dataset = graph_list[:val_size]
    train_dataset = graph_list[val_size:]

    print("The size of the predictor's dataset is :", len(graph_list))
    print(f" The shape of Predictor's training dataset is :{train_dataset[0].x.shape}")
    print(f"The size of predictor's training dataset is: {len(train_dataset)}")
    print(f"The size of predictor's validation dataset is: {len(val_dataset)}")

    start_train_time = time.time()

    if config["param"]["encoding_method"] == "embedding":

        feature_size = int(config["param"]["nfcode"]) + int(config["param"]["noptioncode"])

    else:

        if config["param"]["feature_size_choice"] == "total_functions":
            feature_size = int(config["param"]["total_function"])

        if config["param"]["feature_size_choice"] == "total_choices":
            feature_size = int(config["param"]["total_choices"])

    model = Predictor(feature_size, dim, 1, drop_out)
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=Batch_Size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd,amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True)

    best_loss = 99999999
    c=0
    print("starting training the predictor ...")

    for e,epoch in tqdm(enumerate(range(num_epoch)),total=num_epoch):
        loss = trainpredictor(model, train_loader,optimizer,e)

        RMSE_train, pearson_tr, kendall_tr, spearman_tr = testpredictor(model,
                                                                        train_loader,title="Predictor training test")
        if e % 5 == 0:

            print(f"Epoch {e}: Loss = {loss}  | RMSE= {RMSE_train}  | PCC = {pearson_tr}")
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "predictor.pth")
        scheduler.step(RMSE_train)


    RMSE_train, pearson_tr, kendall_tr, spearman_tr = testpredictor(model,
                                                                    train_loader,
                                                                    title="Predictor training test")

    add_config("results", "RMSE_train", RMSE_train)
    add_config("results", "pearson_train", pearson_tr)
    add_config("results", "kendall_train", kendall_tr)
    add_config("results", "spearman_train", spearman_tr)

    RMSE_val, pearson_val, kendall_val, spearman_val = testpredictor(model,
                                                                     val_loader,
                                                                     title="Predictor validation test")

    predictor_training_time = round(time.time() - start_train_time, 2)
    add_config("time", "predictor_training_time", predictor_training_time)
    add_config("results", "RMSE_val", RMSE_val)
    add_config("results", "pearson_val", pearson_val)
    add_config("results", "kendall_val", kendall_val)
    add_config("results", "spearman_val", spearman_val)

    sample_list = random_sample(e_search_space=e_search_space, n_sample=predict_sample, predictor=True)

    lists = [elt for elt in range(0, len(sample_list), int(config["param"]["batch_sample"]))]
    TopK_models = []
    start_predict_time = time.time()
    print("Start prediction...")
    edge_index = get_edge_index(sample_list[0])
    for i in tqdm(lists, total=len(lists)):
        a = i + int(config["param"]["batch_sample"])

        if a > len(sample_list):
            a = len(sample_list)

        sample = sample_list[i:a]

        #    transform model configuration into graph data
        graph_list = []
        for model_config in sample:
            x = get_nodes_features(model_config, e_search_space)

            graphdata = Data(x=x,
                             edge_index=edge_index,
                             num_nodes=x.shape[0],
                             model_config_choices=deepcopy(model_config))
            graph_list.append(graphdata)

        sample_dataset = DataLoader(graph_list, batch_size=Batch_Size, shuffle=False)

        Trained_model = Predictor(feature_size, dim, 1, drop_out).to(device)
        Trained_model.load_state_dict(torch.load("predictor.pth"))

        TopK = predict_accuracy_using_graph(Trained_model, sample_dataset)
        TopK_models.append(TopK)

    TopK_model = pd.concat(TopK_models)
    TopK_model = TopK_model.nsmallest(k, search_metric, keep="all")
    TopK_final = TopK_model[:k]
    print(TopK_final)
    prediction_time = round(time.time() - start_predict_time, 2)
    print("\n End architecture performance prediction. ")
    add_config("time", "pred_time", prediction_time)

    return TopK_final


def get_prediction_from_table(performance_record, e_search_space):
    set_seed()
    predict_sample = int(config["param"]["predict_sample"])
    search_metric = config["param"]["search_metric"]
    k = int(config["param"]["k"])
    start_time = time.time()
    lb_make = LabelEncoder()
    performance_record = pd.read_csv(performance_record)
    df = performance_record
    x = df.iloc[:, :-1]
    y = df["RMSE"]

    # if config["param"]["encoding_method"] =="embedding":
    #         x =round((x-x.mean())/x.std(),5)

    for col in x.columns:
        if col in ['gnnConv1', 'gnnConv2',
                   "aggregation1", "aggregation2",
                   'normalize1', 'normalize2',
                   'activation1', 'activation2',
                   'pooling', 'criterion', "optimizer"]:

            x[col] = x[col].astype("category")
            if config["param"]["encoding_method"] == "one_hot":
                x[col] = lb_make.fit_transform(x[col])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    start_train_time = time.time()
    # regr1 = MLPRegressor(random_state=1, max_iter=2000).fit(X_train, np.ravel(y_train))
    regr1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300).fit(X_train,
                                                                                        np.ravel(y_train, order='C'))

    r2_tr1, pearson_tr1, kendall_tr1, spearman_tr1 = evaluate_model_predictor(y_test, regr1.predict(X_test),
                                                                              "AdaBoostRegressor")
    df1 = pd.DataFrame(list(zip(y_train, regr1.predict(X_test))), columns=['real_acc', 'predicted_acc'])
    print(
        f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}")

    regr2 = RandomForestRegressor(n_estimators=300).fit(X_train, np.ravel(y_train))
    r2_tr2, pearson_tr2, kendall_tr2, spearman_tr2 = evaluate_model_predictor(y_test, regr2.predict(X_test),
                                                                              "RandomForestRegressor")
    df2 = pd.DataFrame(list(zip(y_train, regr2.predict(X_test))), columns=['real_acc', 'predicted_acc'])
    print(
        f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}")
    # r2_1,r2_3=0,0

    regr3 = MLPRegressor(random_state=1, max_iter=3000).fit(X_train, np.ravel(y_train))
    r2_tr3, pearson_tr3, kendall_tr3, spearman_tr3 = evaluate_model_predictor(y_test, regr3.predict(X_test),
                                                                              "MLPRegressor")
    df3 = pd.DataFrame(list(zip(y_train, regr3.predict(X_test))), columns=['real_acc', 'predicted_acc'])
    print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")

    regr4 = SGDRegressor().fit(X_train, np.ravel(y_train))
    r2_tr4, pearson_tr4, kendall_tr4, spearman_tr4 = evaluate_model_predictor(y_test, regr4.predict(X_test), "LassoCV")
    df4 = pd.DataFrame(list(zip(y_train, regr4.predict(X_test))), columns=['real_acc', 'predicted_acc'])
    print(f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}")

    with open("regression_result1.doc", "w") as regr_:
        regr_.write("test set result")
        regr_.write(
            f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}\n")
        regr_.write(
            f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}\n")
        regr_.write(
            f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}\n")
        regr_.write(
            f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}\n")

    regr1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300).fit(X_train,
                                                                                        np.ravel(y_train, order='C'))

    r2_tr1, pearson_tr1, kendall_tr1, spearman_tr1 = evaluate_model_predictor(y_train, regr1.predict(X_train),
                                                                              "AdaBoostRegressor")
    # df1 = pd.DataFrame(list(zip(y_train,  regr1.predict(X_test))), columns =['real_acc', 'predicted_acc'])
    print(
        f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}")

    regr2 = RandomForestRegressor(n_estimators=200).fit(X_train, np.ravel(y_train))
    r2_tr2, pearson_tr2, kendall_tr2, spearman_tr2 = evaluate_model_predictor(y_train, regr2.predict(X_train),
                                                                              "RandomForestRegressor")
    print(
        f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}")

    regr3 = MLPRegressor(random_state=1, max_iter=2000).fit(X_train, np.ravel(y_train))
    r2_tr3, pearson_tr3, kendall_tr3, spearman_tr3 = evaluate_model_predictor(y_train, regr3.predict(X_train),
                                                                              "MLPRegressor")
    print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")

    regr4 = LassoCV(cv=8).fit(X_train, np.ravel(y_train))
    r2_tr4, pearson_tr4, kendall_tr4, spearman_tr4 = evaluate_model_predictor(y_train, regr4.predict(X_train),
                                                                              "LassoCV")
    print(f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}")

    with open("regression_result2.doc", "w") as regr_:
        regr_.write("train set result")
        regr_.write(
            f"regression with RandomForestRegressor gives R2_score={r2_tr2}|r={pearson_tr2}|rho={spearman_tr2}|tau={kendall_tr2}\n")
        regr_.write(
            f"regression with AdaboostRegressor gives R2 score={r2_tr1}|r={pearson_tr1}|rho={spearman_tr1}|tau={kendall_tr1}\n")
        regr_.write(
            f"regression with MLP gives R2 score={r2_tr3}|r={pearson_tr3}|rho={spearman_tr3}|tau={kendall_tr3}\n")
        regr_.write(
            f"regression with SGD gives R2 score={r2_tr4}|r={pearson_tr4}|rho={spearman_tr4}|tau={kendall_tr4}\n")

    regr = regr2
    RMSE_train = r2_tr2
    pearson_tr = pearson_tr2
    kendall_tr = kendall_tr2
    spearman_tr = spearman_tr2
    regressor = "RandomForestRegressor"
    df = df1

    print(f"Selected regressor: {regressor} --R2_Score = {RMSE_train}")

    predictor_training_time = round(time.time() - start_train_time, 2)
    add_config("time", "predictor_training_time", predictor_training_time)
    add_config("time", "predictor_training_time", predictor_training_time)
    add_config("results", "RMSE_trainain", RMSE_train)
    add_config("results", "pearson_train", pearson_tr)
    add_config("results", "kendall_train", kendall_tr)
    add_config("results", "spearman_train", spearman_tr)

    # ----- Predire la performnance de tous les models present dans le search space
    print("start prediction.")
    start_predict_time = time.time()
    sample_list = random_sample(e_search_space=e_search_space, n_sample=predict_sample, predictor=True)
    print(f"Predicting performance for {len(sample_list)} architecture...")
    lists = range(0, len(sample_list), int(config["param"]["batch_sample"]))
    TopK_model = []

    for i in tqdm(lists,total=len(lists)):
        a = i + int(config["param"]["batch_sample"])
        if a > len(sample_list):
            a = len(sample_list) - 1

        sample = sample_list[i:a]

        #
        predictor_dataset = defaultdict(list)
        for model_config in sample:
            for function, option in model_config.items():
                if config["param"]["encoding_method"] == "one_hot":
                    predictor_dataset[function].append(option[0])
                elif config["param"]["encoding_method"] == "embedding":
                    predictor_dataset[function].append(option[2])

        df = pd.DataFrame.from_dict(predictor_dataset)
        df = df.sample(frac=1).reset_index(drop=True)
        df_temp = df.copy(deep=True)
        # if config["param"]["encoding_method"] =="embedding":
        #    df_temp =round((df_temp-df_temp.mean())/df_temp.std(),5)

        for col in df_temp.columns:
            if col in ['gnnConv1', 'gnnConv2', "aggregation1", "aggregation2", 'normalize1', 'normalize2',
                       'activation1', 'activation2', 'pooling', 'criterion',
                       "optimizer"]:  # some function sets do not required do be categorized such as lr, dropout
                df_temp[col] = df_temp[col].astype("category")
                if config["param"]["encoding_method"] == "one_hot":
                    df_temp[col] = lb_make.fit_transform(df_temp[col])

        predicted_accuracy = regr.predict(df_temp)
        df['AUC_PR'] = predicted_accuracy
        TopK = df.nlargest(k, 'AUC_PR', keep="all")
        TopK = TopK[:k]
        TopK_model.append(TopK)

    TopK_models = pd.concat(TopK_model)

    TopK_models = TopK_models.nlargest(k, 'AUC_PR', keep="all")
    TopK_final = TopK_models[:k].sample(frac=1).reset_index(drop=True)
    prediction_time = round(time.time() - start_predict_time, 2)
    print(TopK_final)
    add_config("time", "pred_time", prediction_time)
    print(" End of prediction")
    return TopK_final


def evaluate_model_predictor(y_true0, y_pred0, title="Predictor training"):
    dataset_name = config["dataset"]["dataset_name"]

    lst = []
    y_pred = []
    y_true = []

    for i in range(len(y_pred0)):

        if y_pred0[i] != 0:
            y_true.append(y_true0[i])
            y_pred.append(y_pred0[i])
            lst.append([y_true0[i], y_pred0[i]])
    if len(y_true) < 2:
        y_true = y_true0
        y_pred = y_pred0
    df = pd.DataFrame(lst, columns=['True', 'Predict'])
    df.to_csv(f'{config["path"]["plots_folder"]}/{title}_{dataset_name}.csv')

    # print(f"true value = {y_true} and predicted value is {y_pred}")
    # R2_Score=round(math.sqrt(mean_squared_error(y_true,  y_pred)),2)
    kendalltau = round(stats.kendalltau(y_true, y_pred)[0], 10)
    spearmanr = round(stats.spearmanr(y_true, y_pred)[0], 10)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = round(math.sqrt(MSE), 10)
    pearson = round(stats.pearsonr(y_true, y_pred)[0], 10)

    if title != "Sampling_distribution":
        if title == "Predictor training test":
            col = "red"
        elif title == "Predictor validation test":
            col = "dodgerblue"
        elif title == "Predictor evaluation test":
            col = "limegreen"
        else:
            col = "red"

        # Visualising the Test set results

        # nb=np.array([i for i in range(min1,min1)])
        plt.figure(figsize=(8, 8))

        a = min(y_pred)
        b = min(y_true)

        xmin = min(min(y_pred), min(y_true))
        xmax = max(max(y_pred), max(y_true))

        lst = [xmin, xmax]
        # lst =[a for a in range(0,100)]
        plt.plot(lst, lst, color='black', linewidth=0.6)
        plt.scatter(y_true, y_pred, color=col, linewidth=0.8)

        # plt.title(f'(r={round(pearson,2)},rho={round(spearmanr,2)},tau={kendalltau})',y=1.02,size=28)#,R2_Score={R2_Score}
        plt.xlabel(f'True RMSE', fontsize=28)
        plt.ylabel(f'Predicted RMSE', fontsize=28)  # ,fontweight = 'bold'
        # plt.legend()
        plt.grid()
        # plt.show()
        plt.savefig(f'{config["path"]["plots_folder"]}/{title}_{dataset_name}.pdf', bbox_inches="tight", dpi=300)
    return RMSE, pearson, kendalltau, spearmanr


def compute_metrics(y_true, y_pred):
    performance = {}
    # print([a for a in y_true])
    # print([b for b in y_pred])
    if "classification" in config["dataset"]["type_task"]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = auc(recall, precision)

        auc_roc = roc_auc_score(y_true, y_pred)
        performance["acc"] = acc
        performance["f1score"] = f1
        performance["mcc"] = mcc
        performance["auc_pr"] = auc_pr
        performance["auc_roc"] = auc_roc

    elif "regression" in config["dataset"]["type_task"]:
        kendall, _ = kendalltau(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
        MSE = mean_squared_error(y_true, y_pred)
        RMSE = round(math.sqrt(MSE), 10)
        pcc, _ = pearsonr(y_true, y_pred)
        performance["kendalltau"] = kendall
        performance["spearmanr"] = spearman
        performance["RMSE"] = RMSE
        performance["pcc"] = pcc

    return performance
