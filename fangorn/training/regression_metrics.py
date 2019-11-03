def mae(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def msle(y_true, y_pred):
    from sklearn.metrics import mean_squared_log_error
    return mean_squared_log_error(y_true, y_pred)

def rmsle(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_log_error
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def mdae(y_true, y_pred):
    from sklearn.metrics import median_absolute_error
    return median_absolute_error(y_true, y_pred)

def r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)