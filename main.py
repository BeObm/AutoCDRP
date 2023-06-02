# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:53:16 2021

@author: Mo
"""


import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

from  search_space_manager.search_space import *
from search_space_manager.sample_models import *
from search_algo.PCRS import *
from search_space_manager.sample_models import *
from search_algo.predictor_model import *
from search_algo.write_results import *
from search_algo.get_test_acc import *
from load_data.load_data import *
from search_algo.utils import manage_budget,Generate_time_cost
from datetime import date
import random
import time
from settings.config_file import *
import argparse

if __name__ == "__main__":
    set_seed()
    torch.cuda.empty_cache()


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Dataset name", default="CCLE")
    parser.add_argument("--type_task" , help="type_task", default="graph regression")
    parser.add_argument("--search_metric", type=str, default="RMSE", help="metric for search guidance",choices=["spearmanr","pcc","auc_pr","mcc","f1score","auc_roc","RMSE"])

    parser.add_argument("--experiment", type=str, default="cell_blind", help="type of experiment") # "cell_blind", "drug_blind","mix"
    args = parser.parse_args()

    create_config_file(args.dataset_name,args.experiment,args.type_task)

    # add_config("dataset", "dataset_root", f"{project_root_dir}/data/{config['dataset']['dataset_name']}")
    add_config("dataset", "dataset_name", args.dataset_name)
    add_config("param", "search_metric", args.search_metric)
    add_config("dataset", "type_experiment", args.experiment)
    manage_budget()
    create_paths(args.dataset_name,args.experiment,args.type_task)

    torch.cuda.empty_cache()

    e_search_space,option_decoder = create_e_search_space()
    total_search_timestart = time.time()

    performance_records_path = get_performance_distributions(e_search_space)

    TopK_final = get_prediction(performance_records_path,e_search_space,config["predictor"]["predictor_type"])

    best_model= get_best_model(TopK_final,option_decoder)
    performances= Evaluate_best_model(best_model)

    total_search_time = round(time.time() - total_search_timestart,2)
    add_config("time","total_search_time",total_search_time)

    write_results(best_model,performances)