from typing import Any, Callable, Dict, List, Tuple, Union
import os
from configparser import ConfigParser

import pandas as pd

from config import read_conf

def get_info_from_file(filename):
    ''' Get all information {attribute = value} pairs from the public.info file'''
    info = {}
    with open (filename, "r") as info_file:
        lines = info_file.readlines()
        features_list = list(map(lambda x: tuple(x.strip("\'").split(" = ")), lines))

        for (key, value) in features_list:
            info[key] = value.rstrip().strip("'").strip(' ')
            if info[key].isdigit(): # if we have a number, we want it to be an integer
                info[key] = int(info[key])
    return info     


def read_prepare_data(dataset_to_work: str) -> List[pd.DataFrame]:
    """
    Read dataset from automl challenge returnig pandas dataframe
    """
    
    def _label_powerset(y_train: pd.DataFrame, inv_map: dict = None):
        """
        Execute a label powerset Transformation
        """
        y_train['tmp_all'] = y_train.astype(str).sum(axis=1)

        if not inv_map:
            map_values = pd.DataFrame(y_train['tmp_all'].value_counts()).reset_index().drop(['tmp_all'], axis=1).to_dict()['index']
            inv_map = {v: k for k, v in map_values.items()}

        y_train['class'] = y_train['tmp_all'].map(inv_map)
        return y_train[['class']], inv_map

    config_parser = config_parser = read_conf.read_conf_file()  
    # caminho base dos arquivos baixados
    data_path = dict(config_parser.items('ML_CHALLENGE_DATA_PATH'))['path']
    # datasets baixados

    dataset_path = (f"{data_path}/{dataset_to_work}/{dataset_to_work}")
    valid_path = (f"{data_path}/valid_solution/{dataset_to_work}")
    info_dict = get_info_from_file(dataset_path+"_public.info")
    
    task = info_dict['task']

    X_all = pd.read_csv(dataset_path+"_train.data", sep=" ", header=None, usecols=[i for i in range(0,info_dict['feat_num'])] )
    y_all =  pd.read_csv(dataset_path+"_train.solution", sep=" ", header=None, usecols=[i for i in range(0,info_dict['target_num'])] )

#     X_valid = pd.read_csv(dataset_path+"_valid.data", sep=" ", header=None, usecols=[i for i in range(0,info_dict['feat_num'])] )
#     y_valid = pd.read_csv(valid_path+"_valid.solution", sep=" ", header=None, usecols=[i for i in range(0,info_dict['target_num'])] )

    if task == 'multilabel.classification':
        # transforma em problema multiclasse
        # conta os valroes de cada classe, transforma em um dict, e troca os indices do dict para mapearmos cada classe (Label Powerset)
        y_all, inv_map = _label_powerset(y_train)
#         y_valid, _ = _label_powerset(y_valid, inv_map) 
        
    return X_all, y_all