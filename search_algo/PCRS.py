# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:37:23 2021

@author: Mo
"""

import time
import pandas as pd
import statistics as stat
import copy
from collections import defaultdict
from search_algo.utils import *
from torch_geometric.data import Data
from search_algo.predictor_model import *
from search_space_manager.sample_models import *
from load_data.load_data import *
from search_space_manager.search_space import *
from search_space_manager.map_functions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

import importlib
from settings.config_file import *
import time

set_seed()


def get_performance_distributions(e_search_space):  # get performance distribution of s*n models (n = search space size)
    set_seed()
    torch.manual_seed(num_seed)
    best_loss_param_path = f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"
    search_metric = config["param"]["search_metric"]
    type_task = config["dataset"]["type_task"]
    epochs = int(config["param"]["sample_model_epochs"])
    n_sample = int(config["param"]["N"])
    model_list = sample_models(n_sample, e_search_space)
    timestart = time.time()  # to record the total time to get the performance distribution set

    train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"])
    gcn, train_model, test_model = get_train(type_task)

    edge_index = get_edge_index(model_list[0])
    # print("example of model_config", model_list[0])
    predictor_dataset = defaultdict(list)
    graph_list = []
    if search_metric == "RMSE":
        best_performance = 99999
    else:
        best_performance = 0
    print(f' \n Constructing a dataset consisting of {n_sample} models for predictor training. \n')

    for no, submodel in tqdm(enumerate(model_list),total=len(model_list)):

        model, criterion, optimizer = get_model_instance(submodel, gcn)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        for epoch in range(epochs):
            train_model(model, train_loader, criterion, optimizer, epoch + 1)
            performances = test_model(model, val_loader)
            performance = performances[search_metric]
            scheduler.step(performance)

        start_time = time.time()
        performances = test_model(model, val_loader)
        performance = performances[search_metric]
        test_time = time.time() - start_time
        if math.isnan(performance):
            test_accc = 0

        if (performance < best_performance and search_metric == "RMSE") or (
                performance > best_performance and search_metric != "RMSE"):
            best_performance = performance
            best_sample = copy.deepcopy(submodel)
            best_sample[search_metric] = performance
            print(
                f'{[a[0] for a in submodel.values()]}-->  {search_metric} = {round(performance, 4)}  ===++ Actual Best Performance')
            for k, v in performances.items():
                print(f"{k} = {round(v, 4)}")
        else:
            print(f'{[a[0] for a in submodel.values()]} -->  {search_metric} = {round(performance, 4)} ')

        #  transform model configuration into graph data
        if (config["param"]["predictor_dataset_type"]) == "graph":
            # edge_index=get_edge_index(model_list[0])
            x = get_nodes_features(submodel, e_search_space)
            y = np.array(performance)

            y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            graphdata = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0],
                             model_config_choices=deepcopy(submodel))
            graph_list.append(graphdata)
            torch.save(graphdata, f"{config['path']['predictor_dataset_folder']}/graph{no + 1}_{x.shape[1]}Feats.pt")

        elif (config["param"]["predictor_dataset_type"]) == "table":
            for function, option in submodel.items():
                if config["param"]["encoding_method"] == "one_hot":
                    predictor_dataset[function].append(option[0])
                elif config["param"]["encoding_method"] == "embedding":
                    predictor_dataset[function].append(option[2])
            predictor_dataset[search_metric].append(performance)

    sample_time = round(time.time() - timestart, 2)
    add_config("time", "distribution_time", sample_time)
    add_config("results", "best_sample_Accuracy", best_performance)
    if (config["param"]["predictor_dataset_type"]) == "graph":
        return config['path']['predictor_dataset_folder']
    if (config["param"]["predictor_dataset_type"]) == "table":
        df = pd.DataFrame.from_dict(predictor_dataset, orient="columns")
        dataset_file = f'{config["path"]["predictor_dataset_folder"]}/{config["dataset"]["dataset_name"]}-{config["param"]["budget"]} samples.csv'
        df.to_csv(dataset_file)
        return dataset_file


def get_best_model(topk_list, option_decoder):
    set_seed()
    torch.cuda.empty_cache()
    torch.manual_seed(num_seed)
    best_loss_param_path = f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"
    search_metric = config["param"]["search_metric"]
    pred_Accuracy = []
    real_Accuracy = []
    type_task = config["dataset"]["type_task"]
    z_topk = int(config["param"]["z_topk"])
    epochs = int(config["param"]["topk_model_epochs"])

    start_time = time.time()
    train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"])
    task_model, train_model, test_model = get_train(type_task)

    min_rmse = 99999999999

    print("Training tok k models ...")

    Y = 999999
    for filename in glob.glob(config["path"]["predictor_dataset_folder"] + '/*'):
        data = torch.load(filename)
        data.y = data.y.view(-1, 1)

        if data.y.item() < Y:
            Y = data.y.item()
            submodel = deepcopy(data.model_config_choices)

        bestmodel = copy.deepcopy(submodel)

        for k, v in bestmodel.items():
            if k != search_metric:
                bestmodel[k] = v[0]
    print(f"Best sample {search_metric} = {Y}")
    for idx, row in tqdm(topk_list.iterrows(),total=len(topk_list)):
        dict_model = {}  #

        if (config["param"]["predictor_dataset_type"]) == "graph":
            for choice in row["model_config"]:
                dict_model[choice[0]] = option_decoder[choice[1]]

        elif (config["param"]["predictor_dataset_type"]) == "table":
            for function in topk_list.columns:
                if function != search_metric:
                    if config["param"]["encoding_method"] == "one_hot":
                        dict_model[function] = row[function]
                    elif config["param"]["encoding_method"] == "embedding":
                        dict_model[function] = option_decoder[row[function]]

        model, criterion, optimizer = get_model_instance2(dict_model, task_model)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

        try:
            model.load_state_dict(best_loss_param_path)

        except:
            pass

        Accuracy_list = []

        for i in range(z_topk):
            best_loss = 999999999999999999

            for epoch in range(epochs):
                loss = train_model(model, train_loader, criterion, optimizer)
                performance = test_model(model, val_loader)[search_metric]
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), best_loss_param_path)
                scheduler.step(performance)

            best_model, criterion, optimizer = get_model_instance2(dict_model, task_model)
            best_model.load_state_dict(torch.load(best_loss_param_path))

            performance = test_model(best_model, val_loader)[search_metric]
            Accuracy_list.append(performance)
        val_Accuracy = stat.mean(Accuracy_list)

        pred_Accuracy.append(row[search_metric])
        real_Accuracy.append(val_Accuracy)

        if min_rmse > val_Accuracy:
            min_rmse = val_Accuracy
            bestmodel = copy.deepcopy(dict_model)
    best_acc_time = round(time.time() - start_time, 2)
    add_config("time", "best_acc_time", best_acc_time)

    RMSE, pearson, kendalltau, spearmanr = evaluate_model_predictor(real_Accuracy, pred_Accuracy,
                                                                    title="Predictor evaluation")
    add_config("results", "RMSE_Score_test", RMSE)
    add_config("results", "pearson_test", pearson)
    add_config("results", "kendall_test", kendalltau)
    add_config("results", "spearman_test", spearmanr)

    return bestmodel


def get_train(type_task):
    if "regression" in config['dataset']['type_task']:
        task_model_obj = importlib.import_module(f"GNN_models.graph_regression")
    elif "classification" in config['dataset']['type_task']:
        task_model_obj = importlib.import_module(f"GNN_models.graph_classification")
    gcn = getattr(task_model_obj, "GNN_Model")
    train_model = getattr(task_model_obj, "train_function")
    test_model = getattr(task_model_obj, "test_function")
    return gcn, train_model, test_model


def get_model_instance(submodel, GCN):
    set_seed()
    type_task = config["dataset"]["type_task"]
    param_dict = {}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1'] = submodel['aggregation1'][0]
    param_dict['aggregation2'] = submodel['aggregation2'][0]
    param_dict["normalize1"] = map_normalization(submodel["normalize1"][0])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"][0])
    param_dict["dropout"] = submodel["dropout"][0]
    param_dict["multi_head1"] = submodel["multi_head1"][0]
    param_dict["multi_head2"] = submodel["multi_head2"][0]
    param_dict['activation1'] = map_activation_function(submodel['activation1'][0])
    param_dict['activation2'] = map_activation_function(submodel['activation2'][0])
    if "graph" in config["dataset"]['type_task']:
        param_dict["global_pooling"] = map_pooling(submodel['pooling'][0])
    param_dict['type_task'] = type_task
    param_dict["gnnConv1"] = map_gnn_model(submodel['gnnConv1'][0])
    param_dict["gnnConv2"] = map_gnn_model(submodel['gnnConv2'][0])
    param_dict["hidden_channels"] = submodel['hidden_channels'][0]

    model = GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"] == "yes":
        model = nn.DataParallel(model)
    model.to(device)

    criterion = map_criterion(submodel['criterion'][0])
    optimizer = map_optimizer(submodel['optimizer'][0], model, submodel['lr'][0], submodel['weight_decay'][0])

    return model, criterion, optimizer


def get_model_instance2(submodel, GCN):
    """
    Parameters
    ----------
    submodel : TYPE  dictionnary comprising component sampled from search space
        DESCRIPTION.

    Returns
    -------
    None.
 
    """
    set_seed()
    type_task = config["dataset"]["type_task"]
    param_dict = {}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1'] = submodel['aggregation1']
    param_dict['aggregation2'] = submodel['aggregation2']
    param_dict["normalize1"] = map_normalization(submodel["normalize1"])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"])
    param_dict["dropout"] = submodel["dropout"]
    param_dict["multi_head1"] = submodel["multi_head1"]
    param_dict["multi_head2"] = submodel["multi_head2"]
    param_dict['activation1'] = map_activation_function(submodel['activation1'])
    param_dict['activation2'] = map_activation_function(submodel['activation2'])
    if "graph" in config["dataset"]['type_task']:
        param_dict["global_pooling"] = map_pooling(submodel['pooling'])
    param_dict['type_task'] = type_task
    param_dict["gnnConv1"] = map_gnn_model(submodel['gnnConv1'])
    param_dict["gnnConv2"] = map_gnn_model(submodel['gnnConv2'])
    param_dict["hidden_channels"] = submodel['hidden_channels']

    model = GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"] == "yes":
        model = nn.DataParallel(model)
    torch.cuda.empty_cache()
    model.to(device)

    criterion = map_criterion(submodel['criterion'])
    optimizer = map_optimizer(submodel['optimizer'], model, submodel['lr'], submodel['weight_decay'])

    return model, criterion, optimizer
