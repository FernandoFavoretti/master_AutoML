import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from typing import Any, Callable, Dict, List, Tuple, Union

def extra_trees_feature_selection(X_all: pd.DataFrame, y_all: pd.DataFrame) -> pd.DataFrame:
    """
    Fits entire data and using simple feature importance to select only importante features
    """
    y_all = y_all.values.ravel()
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_all, y_all)
    model = SelectFromModel(clf, prefit=True)
    return pd.DataFrame(model.transform(X_all))