# -*- coding: utf-8 -*-


import torch
import random
import numpy as np
from configparser import ConfigParser
import os.path as osp
import os
from datetime import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 30
num_seed = 1024
config = ConfigParser()
Batch_Size = 32

RunCode = dates = datetime.now().strftime("%d-%m_%Hh%M")

def set_seed(seed=num_seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


# =========== First level of  running configurations  =================>

project_root_dir = os.path.abspath(os.getcwd())



# Second  level of  running configurations
def create_config_file(dataset_name,run_detail,type_task):
    configs_folder = osp.join(project_root_dir, f'results/{type_task}/{dataset_name}/{RunCode}({run_detail})')
    os.makedirs(configs_folder, exist_ok=True)
    config_filename = f"{configs_folder}/ConfigFile_{RunCode}.ini"

    # No neeed to fill dataset information twice
    config["dataset"] = {
        "type_experiment": "mix",  # "cell_blind", "drug_blind","mix"
        "dataset_name": dataset_name,  # Citeseer,
        'type_task': type_task,  # it could be "graph classification", "link prediction",node classification
        "dataset_root": f"{project_root_dir}/data/{dataset_name}"
    }

    # fill other configuration information
    config["param"] = {
        "project_dir": project_root_dir,
        'config_filename': config_filename,
        "run_code": RunCode,
        "budget": 800,
        "k": 100,
        "z_sample": 1,  # Number of time  sampled models are trained before we report their performance
        "z_topk": 1,
        "z_final": 1,
        "nfcode": 56,  # number of digit for each function code when using embedding method
        "noptioncode": 8,
        "sample_model_epochs": 200,
        "topk_model_epochs": 200,
        "best_model_epochs": 300,
        "encoding_method": "one_hot",
        "type_sampling": "controlled_stratified_sampling",  # random_sampling, uniform_sampling, controlled_stratified_sampling
        "predictor_dataset_type": "graph",
        "feature_size_choice": "total_choices",  # total_functions total_choices  # for one hot encoding using graph dataset for predictor, use"total choices
        'type_input_graph': "directed",
        "use_paralell": "yes",
        "learning_type": "supervised",
        "predict_sample": 500000,
        "batch_sample": 10000
    }

    config["predictor"] = {
        "predictor_type":"GAT",
        "dim": 1024,
        "drop_out": 0.3,
        "lr": 0.001,
        "wd": 0,
        "num_epoch": 5000,
        "comit_test": "yes"
    }

    config["time"] = {
        "distribution_time": 00,
        "sampling_time": 00
    }

    with open(config_filename, "w") as file:
        config.write(file)


def add_config(section_, key_, value_, ):
    if section_ not in list(config.sections()):
        config.add_section(section_)
    config[section_][key_] = str(value_)
    filename = config["param"]["config_filename"]
    with open(filename, "w") as conf:
        config.write(conf)


def create_paths(dataset_name,run_detail,type_task):
    # Create here path for recording model performance distribution
    result_folder = osp.join(project_root_dir, f'results/{type_task}/{dataset_name}/{RunCode}({run_detail})')
    os.makedirs(result_folder, exist_ok=True)
    add_config("path", "performance_distribution_folder", result_folder)
    add_config("path", "best_model_folder", result_folder)
    add_config("path", "result_folder", result_folder)
    # Create here path for recording details about the result
    result_detail_folder = osp.join(project_root_dir, f'results/result_details/{type_task}')
    os.makedirs(result_detail_folder, exist_ok=True)
    add_config("path", "result_detail_folder", result_detail_folder)

    # Create here path for saving plots
    plots_folder = osp.join(result_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    add_config("path", "plots_folder", plots_folder)

    # create here path for saving predictor results
    predictor_results_folder = osp.join(result_folder, "predictor_training_data")
    os.makedirs(predictor_results_folder, exist_ok=True)
    add_config("path", "predictor_results_folder", predictor_results_folder)

    add_config("path", "predictor_weight_path", result_folder)
    add_config("path", "predictor_dataset_folder", predictor_results_folder)  #uncomment to build predictor dataset if predictor training dataset is not available
    # add_config("path", "predictor_dataset_folder", "results/graph regression/CCLE/20-04_13h22(final_run)/predictor_training_data")
    # add_config("path", "predictor_dataset_folder", "data/predictor_dataset_blind-cell_stratified")
    # add_config("path", "predictor_dataset_folder", "data/predictor_dataset_mix_uniform")
    # add_config("path", "predictor_dataset_folder", "data/predictor_dataset_mix_stratified")


