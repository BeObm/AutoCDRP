# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:37:23 2021

@author: Mo
"""
# import os

import statistics as stat
from search_space_manager.sample_models import *
from load_data.load_data import *
from search_space_manager.search_space import *
from search_space_manager.map_functions import *
from search_algo.PCRS import *
from GNN_models.graph_regression import *
set_seed()

def Evaluate_best_model(submodel):
    set_seed()
    z_final= int(config["param"]["z_final"])
    type_task =config["dataset"]["type_task"]
    epochs= int(config["param"]["best_model_epochs"])   
    best_loss_param_path =f"{config['path']['performance_distribution_folder']}/best_model_param.pth"
    search_metric = config["param"]["search_metric"]
  
    type_task =config["dataset"]["type_task"]
   
    n_sample =int(config["param"]["N"]) 
    
    timestart = time.time()   # to record the total time to get the performance distribution set


    gcn,train_model,test_model=get_train(type_task)   
    
    train_loader, val_loader, test_loader = load_dataset(config["dataset"]["type_experiment"])
   
      
    model, criterion, optimizer =get_model_instance2(submodel,gcn)

    best_model_training_record=defaultdict(list)

    best_loss =99999999
    for epoch in range(epochs):
       train_loss = train_model(model,train_loader, criterion, optimizer,epoch+1)
       train_performances = test_model(model, test_loader)

       best_model_training_record["epoch"].append(epoch)
       best_model_training_record["train_loss"].append(train_loss)
       for k,v in train_performances.items():
           best_model_training_record[k].append(v)

       if train_loss < best_loss:
              best_loss = train_loss
              torch.save(model.state_dict(),best_loss_param_path )
    df = pd.DataFrame.from_dict(best_model_training_record, orient="columns")
    best_model_training_record_file = f'{config["path"]["result_folder"]}/best_model_training_record.csv'
    df.to_csv(best_model_training_record_file)

    # # torch.save(model.state_dict(), f'{config["path"]["best_model_folder"]}/temp_model_dict.pth')
    best_model, criterion, optimizer =get_model_instance2(submodel,gcn)
    best_model.load_state_dict(torch.load(best_loss_param_path))

    performances= test_model(best_model, test_loader)
    for metric, val in performances.items():
        add_config("results",f"AutoCDRP_{metric}",val)
         