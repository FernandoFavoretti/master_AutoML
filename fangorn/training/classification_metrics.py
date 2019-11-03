def acc(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    y_pred = list(map(lambda k: 0 if k<=0.5 else 1, y_pred))
    return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    y_pred = list(map(lambda k: 0 if k<=0.5 else 1, y_pred))
    return f1_score(y_true, y_pred)

def precision(y_true, y_pred):
    from sklearn.metrics import precision_score
    y_pred = list(map(lambda k: 0 if k<=0.5 else 1, y_pred))
    return precision_score(y_true, y_pred, average='macro') 

def recall(y_true, y_pred):
    from sklearn.metrics import recall_score
    y_pred = list(map(lambda k: 0 if k<=0.5 else 1, y_pred))
    return recall_score(y_true, y_pred, average='macro')

def auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)