# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:19:59 2021

@author: Mo
"""
from settings.config_file import *

def write_results(best_model,performances):
    
       k=int(config["param"]["k"])
    
       with open(f'results/result_details/{config["dataset"]["type_task"]}/{config["dataset"]["dataset_name"]}_n{int(config["param"]["budget"])}_K{k}.txt','a+') as results:
            results.write(f'\n #############    Result report    (report time: {RunCode}) seed:{num_seed} #############\n')
            results.write(f'Dataset: {(config["dataset"]["dataset_name"])} \n')
            results.write(f'Type experiment: {config["dataset"]["type_experiment"]} \n')
            results.write(f'size of the search space = : {int(config["param"]["size_sp"])}\n') 
            results.write(f'Encoding method: {(config["param"]["encoding_method"])}\n')  
            results.write(f'Node feature size: {(config["param"]["feature_size_choice"])}\n')  
            results.write(f'Type predictor training graph data: {(config["param"]["type_input_graph"])}\n')  
            results.write(f'Predictor dataset type: {(config["param"]["predictor_dataset_type"])}\n') 
            results.write(f'Type of sampling : {config["param"]["type_sampling"]}\n')  
            results.write(f'Use parallel = : {config["param"]["use_paralell"]}\n\n')  
            results.write(f'Total budget: {int(config["param"]["budget"])}\n')   
            results.write(f'Number of samples: {int(config["param"]["N"])}\n')   
            results.write(f'Number of TopK models: {int(config["param"]["k"])}\n')   
        
            results.write(f'Best model: {best_model}\n')
            # results.write(f'Average training time for each model : {get_minutes(float(config["time"]["distribution_time"])/int(config["param"]["N"]))} minutes\n')
            # results.write(f'Predictor training time: {get_minutes(float(config["time"]["predictor_training_time"]))} minutes\n')
            # results.write(f'all models Prediction time: {get_minutes(float(config["time"]["pred_time"]))} minutes\n')
            # results.write(f'Top-{k} model search time: {get_minutes(float(config["time"]["best_RMSE_time"]))} minutes\n')
            # results.write(f'Total search time: {get_minutes(float(config["time"]["total_search_time"]))} minutes\n\n')
            #
            results.write(f'predictor train RMSE= {float(config["results"]["RMSE_train"])} \n')
            results.write(f'predictor val RMSE= {float(config["results"]["RMSE_val"])} \n')
            # results.write(f'predictor test R2_Score= {float(config["results"]["R2_Score_test"])} \n\n')
            
            results.write(f'predictor train pearson_corr= {float(config["results"]["pearson_train"])} \n')
            results.write(f'predictor val pearson_corr= {float(config["results"]["pearson_val"])} \n')
            # results.write(f'predictor test pearson_corr= {float(config["results"]["pearson_test"])} \n\n')
            
            results.write(f'predictor train kendall_corr= {float(config["results"]["kendall_train"])} \n')
            results.write(f'predictor val kendall_corr= {float(config["results"]["kendall_val"])} \n')
            # results.write(f'predictor test kendall_corr= {float(config["results"]["kendall_test"])} \n\n')
            
            
            results.write(f'predictor train spearman_corr= {float(config["results"]["spearman_train"])} \n')
            results.write(f'predictor val spearman_corr= {float(config["results"]["spearman_val"])} \n\n')
            # results.write(f'predictor test spearman_corr= {float(config["results"]["spearman_test"])} \n\n')
            for metric, val in performances.items():
                 results.write(f'AutoCDRP_{metric}"= {val} \n')


            results.write("-------------------------------------------------\n\n ")
           
       print(f'\n #############    Result report    (report time: {RunCode})  seed:{num_seed} #############\n')
       print(f'Dataset: {(config["dataset"]["dataset_name"])} \n')
       print(f'Type experiment: {config["dataset"]["type_experiment"]} \n')
       print(f'size of the search space = : {int(config["param"]["size_sp"])}\n') 
       print(f'Encoding method: {(config["param"]["encoding_method"])}\n')  
       print(f'Node feature size: {(config["param"]["feature_size_choice"])}\n')  
       print(f'Type predictor training graph data: {(config["param"]["type_input_graph"])}\n')  
       print(f'Predictor dataset type: {(config["param"]["predictor_dataset_type"])}\n') 
       print(f'Type of sampling : {config["param"]["type_sampling"]}\n')  
       print(f'Use parallel = : {config["param"]["use_paralell"]}\n\n')  
       print(f'Total budget: {int(config["param"]["budget"])}\n')   
       print(f'Number of samples: {int(config["param"]["N"])}\n')   
       print(f'Number of TopK models: {int(config["param"]["k"])}\n')   
       print(f'Best model: {best_model}\n')
       # print(f'Average training time for each model : {get_minutes(float(config["time"]["distribution_time"])/int(config["param"]["N"]))} minutes\n')
       # print(f'Predictor training time: {get_minutes(float(config["time"]["predictor_training_time"]))} minutes\n')
       # print(f'all models Prediction time: {get_minutes(float(config["time"]["pred_time"]))} minutes\n')
       # print(f'Top-{k} model search time: {get_minutes(float(config["time"]["best_RMSE_time"]))} minutes\n')
       # print(f'Total search time: {get_minutes(float(config["time"]["total_search_time"]))} minutes\n\n')
       
       print(f'predictor train RMSE= {float(config["results"]["RMSE_train"])} \n')
       print(f'predictor val RMSE= {float(config["results"]["RMSE_val"])} \n')
       # print(f'predictor test R2_Score= {float(config["results"]["R2_Score_test"])} \n\n')
       
       print(f'predictor train pearson_corr= {float(config["results"]["pearson_train"])} \n')
       print(f'predictor val pearson_corr= {float(config["results"]["pearson_val"])} \n')
       # print(f'predictor test pearson_corr= {float(config["results"]["pearson_test"])} \n\n')
        
       print(f'predictor tain kendall_corr= {float(config["results"]["kendall_train"])} \n')
       print(f'predictor val kendall_corr= {float(config["results"]["kendall_val"])} \n')
       # print(f'predictor test kendall_corr= {float(config["results"]["kendall_test"])} \n\n')
        
       print(f'predictor train spearman= {float(config["results"]["spearman_train"])} \n')
       print(f'predictor val spearman= {float(config["results"]["spearman_val"])} \n')
       # print(f'predictor test spearman= {float(config["results"]["spearman_test"])} \n\n')
       for metric, val in performances.items():
           print(f'AutoCDRP_{metric}"= {val} \n')
       print("-------------------------------------------------\n\n ")
           


def get_minutes(seconds):
    return round(seconds/60,2)

def get_hours(seconds):
    return round(seconds/3600,2)