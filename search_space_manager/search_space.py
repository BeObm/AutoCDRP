# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:34:13 2021

@author: Mo
"""
from settings.config_file import *


def create_e_search_space(a=0,b=1):   # a<b
    """
    Function to generate architecture description components

    Parameters
    ----------
    nfcode : TYPE   int
        DESCRIPTION.   number of character to encode the type of function in the search space
    noptioncode : TYPE   int
        DESCRIPTION. number of character to encode a choice of a function in the search space
      
    Returns
    ------
    e_search_space : TYPE  dict
        DESCRIPTION.     enbedded search space

    """
    type_task=config['dataset']["type_task"]
   
    nfcode=int(config["param"]["nfcode"])
    noptioncode=int(config["param"]["noptioncode"])
  
    attention= ["GCNConv","GATConv","linear","GENConv","SGConv","LEConv"]

    agregation=['add',"max","mean"] 
    activation=["PReLU","sigmoid","relu"]
    multi_head= [1,2,4,6]
    hidden_channels =[64,128,256]
    
    sp={}
                                                 
    sp['gnnConv1']=attention
    sp['gnnConv2']= attention
    sp['aggregation1']=agregation 
    sp['aggregation2']=agregation
    sp['activation1']=activation
    sp['activation2']=activation
    sp['multi_head1']=multi_head
    sp['multi_head2']=multi_head

    sp['hidden_channels']= hidden_channels
    sp['dropout']= [0.2, 0.5,0.4,0.3]
    sp['lr']= [1e-4,1e-2, 1e-3]
    sp['weight_decay']=[0,1e-5, 1e-6]
    if "regression" in config['dataset']['type_task']:
        sp['criterion'] = ["MSELoss","smooth_l1_loss"]
    elif "classification" in config['dataset']['type_task']:
        sp['criterion'] = ["CrossEntropyLoss"]
    sp['pooling']=["global_max_pool"]
    sp["optimizer"] = ["adam","sgd"]#
    sp['normalize1'] =["BatchNorm"]
    sp['normalize2'] =["BatchNorm"]

        
    # For quick test the following search space will be used ## MUwech
    total_choices=0    
    t1=1
    max_option=0
    for k,v in sp.items():
        t1=t1*len(v)
        total_choices=total_choices+len(v)
        if len(v)>max_option:
            max_option=len(v)
    add_config("param","max_option",max_option)    
    add_config("param","total_function",len(sp))
    add_config("param","total_choices",total_choices)
    add_config("param","size_sp",t1)
   
  
    print(f'The search space has {len(sp)} functions, a total of {total_choices} choices and {t1} possible GNN models.')
    
    e_search_space,option_decoder = search_space_embeddings(sp,nfcode, noptioncode,a,b)
   
    
    return e_search_space,option_decoder


def create_e_search_space0(a=0, b=1):  # a<b
    """
    Function to generate architecture description components

    Parameters
    ----------
    nfcode : TYPE   int
        DESCRIPTION.   number of character to encode the type of function in the search space
    noptioncode : TYPE   int
        DESCRIPTION. number of character to encode a choice of a function in the search space

    Returns
    ------
    e_search_space : TYPE  dict
        DESCRIPTION.     enbedded search space

    """
    type_task = config['dataset']["type_task"]

    nfcode = int(config["param"]["nfcode"])
    noptioncode = int(config["param"]["noptioncode"])

    attention = ["GCNConv", "GATConv", "linear", "GENConv", "SGConv"]

    agregation = ['add', "max", "mean"]
    activation = ["PReLU", "sigmoid", "relu"]
    multi_head = [1, 2, 4]
    hidden_channels = [64, 128, 256]

    sp = {}

    sp['gnnConv1'] = attention
    sp['gnnConv2'] = attention
    sp['aggregation1'] = agregation
    sp['aggregation2'] = agregation
    sp['activation1'] = activation
    sp['activation2'] = activation
    sp['multi_head1'] = multi_head
    sp['multi_head2'] = multi_head

    sp['hidden_channels'] = hidden_channels
    sp['dropout'] = [0.2, 0.5, 0.4]
    sp['lr'] = [1e-2, 1e-3, 5e-4, 1e-4, 5e-3]
    sp['weight_decay'] = [0,1e-5, 1e-3]
    if "regression" in config['dataset']['type_task']:
        sp['criterion'] = ["MSELoss", "smooth_l1_loss"]
    elif "classification" in config['dataset']['type_task']:
        sp['criterion'] = ["CrossEntropyLoss"]
    sp['pooling'] = ["global_max_pool"]
    sp["optimizer"] = ["adam"]  #
    sp['normalize1'] = ["BatchNorm"]
    sp['normalize2'] = ["BatchNorm"]

    # For quick test the following search space will be used ## MUwech
    total_choices = 0
    t1 = 1
    max_option = 0
    for k, v in sp.items():
        t1 = t1 * len(v)
        total_choices = total_choices + len(v)
        if len(v) > max_option:
            max_option = len(v)
    add_config("param", "max_option", max_option)
    add_config("param", "total_function", len(sp))
    add_config("param", "total_choices", total_choices)
    add_config("param", "size_sp", t1)

    print(f'The search space has {len(sp)} functions, a total of {total_choices} choices and {t1} possible GNN models.')

    e_search_space, option_decoder = search_space_embeddings(sp, nfcode, noptioncode, a, b)

    return e_search_space, option_decoder


def search_space_embeddings(sp,nfcode, noptioncode,a,b):

    i=0
    embeddings_dict={}
    option_decoder={}      # cle= option code, valeur = option
    fcode_list=[]   # list to check duplicate code in function code
         #  liste to check duplicate in option code
    
    for function,options_list in sp.items():  
        embeddings_dict[function]={}
        option_code_list = []

        if config["param"]["encoding_method"] == "embedding":
            if function in ["gnnConv2","activation2","multi_head2","aggregation2","normalize2"]:
                for option in options_list:
                    option_code =i
                    i+=1
                    embeddings_dict[function][option]=(option_code, embeddings_dict[f"{function[:-1]}1"][option][1])

                    option_decoder[option_code]=option

            else:

                    fcode=[random.randint(a, b) for num in range(0, nfcode)]

                   # verifier si une autre fonction na pas le meme code avant de valider le code
                    while fcode in fcode_list:
                        fcode=[random.randint(a, b) for num in range(0, nfcode)]
                    fcode_list.append(fcode)

                    for option in options_list:

                        option_code =i
                        option_encoding=fcode +[random.randint(a, b) for num in range(0, noptioncode)]
                        i+=1
                        while option_encoding in option_code_list:
                            print("option encoding alredy exist")
                            option_encoding = fcode + [random.randint(a, b) for num in range(0, noptioncode)]
                        option_code_list.append(option_encoding)

                        embeddings_dict[function][option]=(option_code,option_encoding)

                         # set decoder dict value for the current option
                        option_decoder[option_code]=option
        else:
                for option in  options_list:
                   option_code =i 
                   i+=1
                   option_encoding= sp[function].index(option)
                   embeddings_dict[function][option]=(option_code,option_encoding)
                   option_decoder[option_code]=option
                   
    
    return embeddings_dict,option_decoder

