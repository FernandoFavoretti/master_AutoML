import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from typing import Any, Callable, Dict, List, Tuple, Union


def make_response_dict(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    make dict for store all splittings
    """
    d_split = {}
    
    d_split['train'] = {}
    d_split['train']['X'] = X_train
    d_split['train']['y'] = y_train
    
    d_split['val'] = {}
    d_split['val']['X'] = X_val
    d_split['val']['y'] = y_val
    
    d_split['test'] = {}
    d_split['test']['X'] = X_test
    d_split['test']['y'] = y_test
    
    return d_split
    

def simple_train_test_val_split(X_all: pd.DataFrame, y_all:pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
    """
    Apply sklearn simple train test split, generating Train Val and Test sets
    """
    X_train, X_test, y_train, y_test  = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    split_dict = make_response_dict(X_train, X_val, X_test, y_train, y_val, y_test)

    return split_dict


def stratified_kfold_train_test_split(X_all: pd.DataFrame, y_all: pd.DataFrame, k: int =10) -> List[Dict[str, Any]]:
    """
    Create a stratified kfold split on dataset,
    return a list with the repsonse dicts for each fold
    """
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    all_kfold = []
    for train_index, test_index in kf.split(X_all, y_all):
        X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = y_all.iloc[train_index], y_all.iloc[test_index]
        all_kfold.append(make_response_dict(X_train, None, X_test, y_train, None, y_test))
    return all_kfold

