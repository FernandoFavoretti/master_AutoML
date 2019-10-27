def acc(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)

def precision(y_true, y_pred):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average=None) 

def recall(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average=None)

def auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)