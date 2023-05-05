import os
import numpy as np
import csv
from pubchempy import *
import math
from rdkit import Chem
import networkx as nx
# from utils import *
import random
import pickle
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
import pandas as pd
from collections import defaultdict
from settings.config_file import *
set_seed()
import requests



def regr_or_class():
    if "regression" in config['dataset']['type_task']:
        return  "regression"
    elif "classification" in config['dataset']['type_task']:
        return "classification"

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='CCLE',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, saliency_map=False):

        # root is required for saving preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            # print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            print("label::",labels)
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.tensor([labels]))

            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target], dtype=torch.float, requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')

        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd


def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True


"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""
# folder = ""

def load_drug_list():
    filename = config["dataset"]["dataset_root"] + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs


def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(config["dataset"]["dataset_root"] + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid

        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(config["dataset"]["dataset_root"] + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)


def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(config["dataset"]["dataset_root"] + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict:
            cid_dict[name] = str(cid)

    unknow_drug = open(config["dataset"]["dataset_root"] + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k: v for k, v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict


def load_cid_dict():
    reader = csv.reader(open(config["dataset"]["dataset_root"] + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k, v in cids_dict.iteritems()]
    inv_cids_dict = {v: k for k, v in cids_dict.iteritems()}
    download('CSV', config["dataset"]["dataset_root"] + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES',
             overwrite=True)
    f = open(config["dataset"]["dataset_root"] + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(config["dataset"]["dataset_root"] + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()


"""
The following code will convert the SMILES format into onehot format
"""


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_drug_smiles(drug_cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{drug_cid}/property/CanonicalSMILES/json"
    response = requests.get(url)
    json_data = response.json()
    smiles = json_data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    return smiles


def smile_to_graph(smile):
    set_seed()
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def get_pubchem_smiles(drug_id):

    pubchem_file= "data/CCLE/CCLE_smiles.csv"
    df =pd.read_csv(pubchem_file)
    for idx,row in df.iterrows():
        if row["pubchem"]==drug_id:
            smiles = row["isosmiles"]
            break

    return smiles

def load_drug_smile():
  set_seed()
  if config["dataset"]["dataset_name"]=="CCLE":
      drug_id_file =config["dataset"]["dataset_root"]+"/drug_name_cid.csv"
      cid_col=2
  elif config["dataset"]["dataset_name"]=="GDSC":
      drug_id_file = config["dataset"]["dataset_root"] + "/drug_feature.csv"
      cid_col = 0

  try:
      smile_df = pd.read_csv(f'{config["dataset"]["dataset_root"]}/drug_smiles.csv')
  except:
      pass

  df = pd.read_csv(drug_id_file)
  drug_dict={}
  drug_smile=[]
  smile_graph={}
  drug_id=0
  save_smile_list=False
  for idx,row in df.iterrows():
      if df.iloc[idx,cid_col] not in drug_dict:
          drug_dict[int(df.iloc[idx,cid_col])]=drug_id
          drug_id+=1
      try:
          smiles=(smile_df.loc[smile_df['drud_cid']==int(df.iloc[idx,cid_col])])['drug_SMILES'].item()
          drug_smile.append(smiles)
      except:
             save_smile_list=True
             drug_smile.append(get_drug_smiles(int(df.iloc[idx,cid_col])))
  smile_graph = {}
  for smile in drug_smile:
      g = smile_to_graph(smile)
      smile_graph[smile] = g


  if save_smile_list==True:
      save_smile=defaultdict(list)
      for i, k in enumerate(list(drug_dict.keys())):
          save_smile["drud_cid"].append(k)
          save_smile["drug_SMILES"].append(drug_smile[i])

      smile_list=pd.DataFrame(save_smile)
      smile_list.to_csv(f"{config['dataset']['dataset_root']}/drug_smiles.csv")

  return drug_dict, drug_smile, smile_graph


def save_cell_mut_matrix():

    if config["dataset"]["dataset_name"]=="CCLE":
        mut_file = config["dataset"]["dataset_root"]+"/cell_mutation.csv"
        cell_dict={}
        df = pd.read_csv(mut_file, index_col=0)
        cell_id=0
        for cell in list(df.index):
            cell_dict[cell]=cell_id
            cell_id+=1
        cell_feature = np.zeros((df.shape[0], df.shape[1]))
        for idx, row in df.iterrows():
            for i,col in enumerate(list(df.columns)):
                cell_feature[cell_dict[idx]-1,i]= row[col]
    elif config["dataset"]["dataset_name"]=="GDSC":

        mut_file1= mut_file = config["dataset"]["dataset_root"]+"/cell_mutation_dim_1_num_0.csv"
        # mut_file2= mut_file = config["dataset"]["dataset_root"]+"/cell_mutation_dim_1_num_1.csv"
        df= pd.read_csv(mut_file1, index_col=0)
        # df2= pd.read_csv(mut_file2, index_col=0)
        # df = pd.concat([df1, df2], axis=1)
        cell_id = 0
        cell_dict = {}
        for cell in list(df.index):
            cell_dict[cell] = cell_id
            cell_id += 1
        cell_feature = np.zeros((df.shape[0], df.shape[1]))
        for idx, row in df.iterrows():
            for i, col in enumerate(list(df.columns)):
                cell_feature[cell_dict[idx] - 1, i] = row[col]

    print(f"The {config['dataset']['dataset_name']} dataset contains {len(cell_dict)}  cells with mutation feature size of {cell_feature.shape[1]}")
    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
def get_cell_drug_response_list():

    if config["dataset"]["dataset_name"]=="GDSC":

        drug_response_file = config["dataset"]["dataset_root"] + "/cell_drug.csv"
        threshold_df = pd.read_csv(config["dataset"]["dataset_root"] + "/threshold.csv",index_col=0)

        df = pd.read_csv(drug_response_file, index_col=0)
        temp_data = []
        for idx, row in df.iterrows():
            for col in df.columns:
                if not pd.isnull(df.loc[idx, col]):
                    if "regression" in config['dataset']['type_task']:
                        ic50 = 1 / (1 + pow(math.exp(float(df.loc[idx, col])), -0.1))
                        temp_data.append((col, idx, ic50))
                    elif "classification" in config['dataset']['type_task']:
                        ic50 = float(df.loc[idx, col])
                        threshold=threshold_df.loc[int(col),"Threshold"]
                        if ic50< threshold:
                             temp_data.append((col, idx, 0))
                        else:
                            temp_data.append((col, idx, 1))

    elif config["dataset"]["dataset_name"]=="CCLE":

        if "regression" in config['dataset']['type_task']:
            drug_response_file = config["dataset"]["dataset_root"] + "/cell_drug.csv"

            df = pd.read_csv(drug_response_file, index_col=0)
            temp_data = []
            for idx, row in df.iterrows():
                for col in df.columns:
                    if not pd.isnull(df.loc[idx, col]):
                        ic50 = 1 / (1 + pow(math.exp(float(df.loc[idx, col])), -0.1))
                        temp_data.append((col, idx, ic50))

        elif "classification" in config['dataset']['type_task']:
            drug_response_file = config["dataset"]["dataset_root"] + "/cell_drug_binary.csv"
            df = pd.read_csv(drug_response_file, index_col=0)
            temp_data = []
            for idx, row in df.iterrows():
                for col in df.columns:
                    if not pd.isnull(df.loc[idx, col]):
                        temp_data.append((col, idx, int(df.loc[idx, col])))


    print(f"The final dataset wil contain {len(temp_data)} records")
    return temp_data

def save_mix_drug_cell_matrix():
    set_seed()

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    # smile_graph
    bExist = np.zeros((len(drug_dict), len(cell_dict)))
    temp_data = get_cell_drug_response_list()
    # temp_data is a list of tuples (drug, cell, IC50)

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []

    random.shuffle(temp_data)
    count_id=0
    for data in temp_data:

        drug, cell, ic50 = data
        # print(f" this is what in record {drug}  | {cell}  |{ic50}")
        if int(drug) in list(drug_dict.keys()) and cell in list(cell_dict.keys()):

            count_id+=1
            xd.append(drug_smile[drug_dict[int(drug)]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[int(drug)], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)

    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)

    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xc_train = xc[:size]
    xc_val = xc[size:size1]
    xc_test = xc[size1:]

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]

    print('preparing ', config["dataset"]["dataset_name"] + config["dataset"]["type_experiment"]+ '_train.pt in pytorch format!')

    train_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_train', xd=xd_train, xt=xc_train,
                                y=y_train,
                                smile_graph=smile_graph)
    val_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                              dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_val', xd=xd_val, xt=xc_val, y=y_val,
                              smile_graph=smile_graph)
    test_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                               dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_test', xd=xd_test, xt=xc_test,
                               y=y_test,
                               smile_graph=smile_graph)


def save_blind_drug_matrix():
    set_seed()
    random.seed(num_seed)
    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()


    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    # xd_unknown = []
    # xc_unknown = []
    # y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    temp_data = get_cell_drug_response_list()  # temp_data is a list of tuples (drug, cell, IC50)
    random.shuffle(temp_data)

    for data in temp_data:
        drug, cell, ic50 = data
        if int(drug) in list(drug_dict.keys()) and cell in list(cell_dict.keys()):
            if int(drug) in dict_drug_cell:
                dict_drug_cell[int(drug)].append((cell, ic50))
            else:
                dict_drug_cell[int(drug)] = [(cell, ic50)]

            bExist[drug_dict[int(drug)], cell_dict[cell]] = 1

    lstDrugTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for drug, values in dict_drug_cell.items():
        pos += 1
        for v in values:
            cell, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstDrugTest.append(drug)

    with open('drug_bind_test', 'wb') as fp:
        pickle.dump(lstDrugTest, fp)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    print('preparing ', config["dataset"]["dataset_name"] + config["dataset"]["type_experiment"]+ '_train.pt in pytorch format!')
    train_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_train', xd=xd_train, xt=xc_train,
                                y=y_train,
                                smile_graph=smile_graph)
    val_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                              dataset= f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_val', xd=xd_val, xt=xc_val, y=y_val,
                              smile_graph=smile_graph)
    test_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                               dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_test', xd=xd_test, xt=xc_test,
                               y=y_test,
                               smile_graph=smile_graph)


def save_blind_cell_matrix():
    # f = open(config["dataset"]["dataset_root"] + "PANCANCER_IC.csv")
    # reader = csv.reader(f)
    # next(reader)
    # random.seed(num_seed)
    set_seed()
    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()



    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []
    #
    # xd_unknown = []
    # xc_unknown = []
    # y_unknown = []

    dict_drug_cell = {}

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    # for item in reader:
    #     drug = item[0]
    #     cell = item[3]
    #     ic50 = item[8]
    #     ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
    #
    #     temp_data.append((drug, cell, ic50))
    temp_data = get_cell_drug_response_list()
    random.shuffle(temp_data)

    for data in temp_data:
        drug, cell, ic50 = data
        if int(drug) in list(drug_dict.keys()) and cell in list(cell_dict.keys()):
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((int(drug), ic50))
            else:
                dict_drug_cell[cell] = [(int(drug), ic50)]

            bExist[drug_dict[int(drug)], cell_dict[cell]] = 1

    lstCellTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    for cell, values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)

    with open('cell_bind_test', 'wb') as fp:
        pickle.dump(lstCellTest, fp)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    print('preparing ', config["dataset"]["dataset_name"] + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_train', xd=xd_train,
                                xt=xc_train, y=y_train,
                                smile_graph=smile_graph)
    val_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                              dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_val', xd=xd_val, xt=xc_val,
                              y=y_val,
                              smile_graph=smile_graph)
    test_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                               dataset=f'{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_test', xd=xd_test, xt=xc_test,
                               y=y_test,
                               smile_graph=smile_graph)

def main(Batch_Size,dataset_size="all"):

    set_seed()
   
    print('\nrunning on ', config["dataset"]["dataset_name"])
    processed_data_file_train =f'{config["dataset"]["dataset_root"]}/processed/{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_train.pt'
    processed_data_file_val = f'{config["dataset"]["dataset_root"]}/processed/{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_val.pt'
    processed_data_file_test = f'{config["dataset"]["dataset_root"]}/processed/{config["dataset"]["dataset_name"]}_{config["dataset"]["type_experiment"]}_{regr_or_class()}_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (
            not os.path.isfile(processed_data_file_test))):
        print('Data set raw Files are missing! Please prepare data set raw and try again')
        exit()
    else:
        train_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                    dataset=config["dataset"]["dataset_name"] +"_"+ config["dataset"]["type_experiment"]+"_"+ regr_or_class()+ '_train')
        val_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                  dataset=config["dataset"]["dataset_name"] +"_"+  config["dataset"]["type_experiment"] + "_"+ regr_or_class()+'_val')
        test_data = TestbedDataset(root=config["dataset"]["dataset_root"],
                                   dataset=config["dataset"]["dataset_name"] +"_"+  config["dataset"]["type_experiment"]+"_"+ regr_or_class()+'_test')
        print(f'train dataset size is {len(train_data)}')
        print(f'val dataset size is {len(val_data)}')
        print(f'test dataset size is {len(test_data)}')


        if dataset_size != "all":
            print(f"original train data size is {len(train_data)} but only {dataset_size} is used")
            train_data = train_data[:dataset_size]

        # make data PyTorch mini-batch processing ready

        train_loader = DataLoader(train_data, batch_size=Batch_Size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=Batch_Size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=Batch_Size, shuffle=False)
        add_config("dataset","len_traindata",len(test_loader))
        print(" Running on GPU ? : ", torch.cuda.is_available())
        return train_loader, val_loader, test_loader

def load_dataset(choice="mix"):
    if choice == "mix":
        # save mix test dataset
        save_mix_drug_cell_matrix()
    elif choice == "drug_blind":
        # save blind drug dataset
        save_blind_drug_matrix()
    elif choice == "cell_blind":
        # save blind cell dataset
        save_blind_cell_matrix()
    else:
        print("Invalide option, wrong type of experiment dataset type ")

    return main(Batch_Size)