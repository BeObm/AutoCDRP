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
from GNN_models.graph_regression import *
from settings.config_file import *
import time
set_seed()



def get_performance_distributions(e_search_space):  # get performance distribution of s*n models (n = search space size)
    set_seed()
    torch.manual_seed(num_seed)
    best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"

    type_task =config["dataset"]["type_task"]
    epochs =int(config["param"]["sample_model_epochs"])
    n_sample =int(config["param"]["N"])
    model_list = sample_models(n_sample, e_search_space)
    timestart = time.time()   # to record the total time to get the performance distribution set

    train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"])
    gcn,train_model,test_model=get_train(type_task)

    edge_index = get_edge_index(model_list[0])
    # print("example of model_config", model_list[0])
    predictor_dataset=defaultdict(list)
    graph_list=[]
    best_AUC_PR = 999
    print(f' \n Constructing a dataset consisting of {n_sample} models for predictor training. \n')

    for no, submodel in tqdm(enumerate(model_list)):


            torch.cuda.empty_cache()
            
            model, criterion, optimizer =get_model_instance(submodel,gcn)

            best_loss = 9999
            
            for epoch in range(epochs):
               start_time = time.time()
               train_loss = train_model(model,train_loader, criterion, optimizer,epoch+1)
               epoch_time = time.time() - start_time
               
            start_time = time.time()
            test_aucpr= round(test_model(model, test_loader)[0],4)
            test_time = time.time() - start_time
            # print(f"testing one epoch costs {test_time}")
            if  math.isnan(test_aucpr):
                test_aucpr= 999
           
            if test_aucpr <= best_AUC_PR:
                best_AUC_PR=test_aucpr
                best_sample=copy.deepcopy(submodel)
                best_sample["AUC_PR"]=test_aucpr
                print(f'-->  AUC_PR = {test_aucpr}  ===++ Actual Best Performance')

            else :
                  print(f'--> AUC_PR = {test_aucpr}')


            #  transform model configuration into graph data
            if (config["param"]["predictor_dataset_type"])=="graph":
                # edge_index=get_edge_index(model_list[0])
                x = get_nodes_features(submodel,e_search_space)
                y=np.array(test_aucpr)
                y=torch.tensor(y,dtype=torch.float32).view(-1,1)
                graphdata=Data(x=x,edge_index =edge_index,y=y,num_nodes=x.shape[0],model_config_choices = deepcopy(submodel))
                graph_list.append(graphdata)
                torch.save(graphdata,f"{config['path']['predictor_dataset_folder']}/graph{no+1}_{x.shape[1]}Feats.pt")

            elif (config["param"]["predictor_dataset_type"])=="table":
                for function,option in submodel.items():
                    if config["param"]["encoding_method"] =="one_hot":
                       predictor_dataset[function].append(option[0])
                    elif config["param"]["encoding_method"] =="embedding":
                       predictor_dataset[function].append(option[2])
                predictor_dataset["AUC_PR"].append(test_aucpr)

                             
                
    sample_time= round(time.time() - timestart,2)
    add_config("time","distribution_time",sample_time)
    add_config("results","best_sample_AUC_PR",best_AUC_PR)
    
    if (config["param"]["predictor_dataset_type"])=="graph":
                 
         return  config['path']['predictor_dataset_folder']
   
    if (config["param"]["predictor_dataset_type"])=="table":
        df =pd.DataFrame.from_dict(predictor_dataset,orient="columns")   
        dataset_file=f'{config["path"]["predictor_dataset_folder"]}/{config["dataset"]["dataset_name"]}-{config["param"]["budget"]} samples.csv'
        df.to_csv(dataset_file)                  
        return dataset_file
 
    
def get_best_model(topk_list,option_decoder):
    set_seed()
    torch.cuda.empty_cache()
    torch.manual_seed(num_seed)
    best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_dist_param.pth"
    
    pred_AUC_PR=[]
    real_AUC_PR=[]
    encoding_method =config["param"]["encoding_method"]
    type_task =config["dataset"]["type_task"]
    z_topk= int(config["param"]["z_topk"])
    epochs= int(config["param"]["topk_model_epochs"])
    n_sample =int(config["param"]["N"])
    type_sampling= config["param"]["type_sampling"]
    start_time = time.time()
    train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"],"not all")
    task_model,train_model,test_model=get_train(type_task)

    min_AUC_PR=999

    print("Training tok k models ...")

    Y = 99
    for filename in glob.glob(config["path"]["predictor_dataset_folder"] + '/*'):
        data = torch.load(filename)
        data.y = data.y.view(-1, 1)

        if data.y.item() < Y:
            Y = data.y.item()
            submodel = deepcopy(data.model_config_choices)

        bestmodel = copy.deepcopy(submodel)

        for k, v in bestmodel.items():
            if k != "AUC_PR":
                bestmodel[k] = v[0]
    print(f"Best sample AUC_PR = {Y}")
    for idx,row in tqdm(topk_list.iterrows()):
        dict_model={}   #
        
        if (config["param"]["predictor_dataset_type"])=="graph":
            for choice in row["model_config"]:
                dict_model[choice[0]]= option_decoder[choice[1]]
    
        elif (config["param"]["predictor_dataset_type"])=="table":
            for function in topk_list.columns: 
                if function !="AUC_PR":
                    if config["param"]["encoding_method"] =="one_hot":
                      dict_model[function]=row[function]
                    elif config["param"]["encoding_method"] =="embedding":
                      dict_model[function]=option_decoder[row[function]]
        
        
        model, criterion, optimizer =get_model_instance2(dict_model,task_model)
        
        # train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"])
        try:
            model.load_state_dict(best_loss_param_path)
           
        except:
             pass
        
        AUC_PR_list=[]

        for i in range(z_topk):
            best_loss=999
            
            for epoch in range(epochs):                                            
               loss = train_model(model,train_loader, criterion, optimizer)
               if loss < best_loss:
                  best_loss = loss
                  torch.save(model.state_dict(),best_loss_param_path )

            best_model, criterion, optimizer =get_model_instance2(dict_model,task_model)
            best_model.load_state_dict(torch.load(best_loss_param_path))          
                    
            val_AUC_PR= test_model(best_model, val_loader)[0]
            AUC_PR_list.append(val_AUC_PR)
        val_AUC_PR = round(stat.mean(AUC_PR_list),4)
       
         
        pred_AUC_PR.append(row["AUC_PR"])
        real_AUC_PR.append(val_AUC_PR)
                 
        if min_AUC_PR > val_AUC_PR:
            min_AUC_PR=val_AUC_PR
            bestmodel=copy.deepcopy(dict_model)
    best_AUC_PR=min_AUC_PR
    best_acc_time = round(time.time() - start_time,2)
    add_config("time","best_acc_time",best_acc_time)
   
    AUC_PR,pearson_test,kendall_test,spearman_test= evaluate_model_predictor(real_AUC_PR,pred_AUC_PR,title="Predictor evaluation")
    # add_config("results","R2_Score_test",R2_Score_test)
    # add_config("results","pearson_test",pearson_test)
    # add_config("results","kendall_test",kendall_test)
    # add_config("results","spearman_test",spearman_test)
     
    return bestmodel       

   
def get_train(type_task):
     return GraphRegression,train_gc,test_gc
             
def get_model_instance(submodel,GCN):
    set_seed()
    type_task =config["dataset"]["type_task"]
    param_dict={}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1']= submodel['aggregation1'][0]
    param_dict['aggregation2']= submodel['aggregation2'][0]
    param_dict["normalize1"] = map_normalization(submodel["normalize1"][0])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"][0])
    param_dict["dropout"] = submodel["dropout"][0]
    param_dict["multi_head1"] = submodel["multi_head1"][0]
    param_dict["multi_head2"] = submodel["multi_head2"][0]
    param_dict['activation1']=map_activation_function(submodel['activation1'][0])
    param_dict['activation2']=map_activation_function(submodel['activation2'][0])
    if "graph" in config["dataset"]['type_task']:
        param_dict["global_pooling"]=map_pooling(submodel['pooling'][0])
    param_dict['type_task'] =type_task
    param_dict["gnnConv1"]=map_gnn_model(submodel['gnnConv1'][0])
    param_dict["gnnConv2"]=map_gnn_model(submodel['gnnConv2'][0])
    param_dict["hidden_channels"]=submodel['hidden_channels'][0]
    
    model=GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes" :  
        model = nn.DataParallel(model)
    model.to(device)

    criterion = map_criterion(submodel['criterion'][0])
    optimizer = map_optimizer(submodel['optimizer'][0] , model, submodel['lr'][0], submodel['weight_decay'][0])
              
    return model, criterion, optimizer

def get_model_instance2(submodel,GCN):
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
    type_task =config["dataset"]["type_task"]
    param_dict={}
    # dist_dict_encoded, sp_dict_decoder = encode_sp(submodel, 'unique_option')    # pour enregitrer just un model    dans le dataset                      
    param_dict['aggregation1']= submodel['aggregation1']
    param_dict['aggregation2']= submodel['aggregation2']
    param_dict["normalize1"] = map_normalization(submodel["normalize1"])
    param_dict["normalize2"] = map_normalization(submodel["normalize2"])
    param_dict["dropout"] = submodel["dropout"]
    param_dict["multi_head1"] = submodel["multi_head1"]
    param_dict["multi_head2"] = submodel["multi_head2"]
    param_dict['activation1']=map_activation_function(submodel['activation1'])
    param_dict['activation2']=map_activation_function(submodel['activation2'])
    if "graph" in config["dataset"]['type_task']:
        param_dict["global_pooling"]=map_pooling(submodel['pooling'])
    param_dict['type_task'] =type_task
    param_dict["gnnConv1"]=map_gnn_model(submodel['gnnConv1'])
    param_dict["gnnConv2"]=map_gnn_model(submodel['gnnConv2'])
    param_dict["hidden_channels"]=submodel['hidden_channels']
    
    model=GCN(param_dict)
    if torch.cuda.device_count() > 1 and config["param"]["use_paralell"]=="yes":
        model = nn.DataParallel(model)
    torch.cuda.empty_cache()
    model.to(device)
    
    criterion = map_criterion(submodel['criterion'])
    optimizer=map_optimizer(submodel['optimizer'] , model, submodel['lr'], submodel['weight_decay']) 

    return model,criterion,optimizer