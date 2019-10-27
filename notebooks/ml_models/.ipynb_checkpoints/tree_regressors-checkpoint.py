import pandas as pd
from typing import Any, Callable, Dict, List, Tuple, Union
from toolz import assoc

def lgbm_regressor(train_set: List[pd.DataFrame],
                         test_set: List[pd.DataFrame],
                         features: List[str],
                         target: str,
                         test_metrics: List[str],
                         validation_set: List[pd.DataFrame] = None,
                         core_params: Dict[str, any] = {},
                         hyper_params: Dict[str, any] = {},
                         prediction_column: str = "preds",
                         real_values_column: str = "real_value",
                         log: bool = False,
                         project_name: str = None
                         ) -> Callable:
    """
    Fits a LGBM regressor to the dataset
    
    Parameters
    ----------
    train_set: List of pandas.DataFrame
        [X_train, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.
        
    test_set: List of pandas.DataFrame
        [X_test, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.

    features: list of str
        the list of features used to train the model
        (used for feature importance and log)
    
    target: str
        the target feature used in the model
        (used for log only)
        
    test_metrics: list of str
        list of measurement metrics to use, must be on
        mae: mean absolute error
        mdae: median absolute error
        mse: mean squared error
        rmse: root mean squared error
        msle: mean squared log error
        rmsle: root mean squared error
        r2: r squared
    
    validation_set: list of [train_data, validation_data]
        Lgbm validation set for earling stop
        
    core_params: dict, optional
        core_params of the algorithm
        defaults
        ---------
        num_boost_round, valid_sets=None, valid_names=None,
        fobj=None, feval=None, init_model=None, feature_name='auto',
        categorical_feature='auto', early_stopping_rounds=None,
        evals_result=None, verbose_eval=True, learning_rates=None,
        keep_training_booster=False, callbacks=None
        ---------
        
    hyperparams: dict, optional
        The params of the lgbm regression
        dict in format of {'hyperparameter_name': hyperparameter_value}
        If not passed the default will be used
        
    prediction_column: str
        The name of the column with the predictions from the model
        default `preds`
        
    real_value: str
        The name of the column with the real value from test set
        default `real_value`
        
    log: boll
        Boolean condition if the model log should be generated into mlflow
    
    project_name: str
        Name of the project for log, only used if log = True
 
    """
    
    import lightgbm as lgbm
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    hyper_params = hyper_params if "objective" in hyper_params else assoc(hyper_params, 'objective', 'regression')
    lgbm_train_set = lgbm.Dataset(X_train, y_train)

    if validation_set:
        lgbm_eval_set = lgbm.Dataset(validation_set[0], validation_set[1])
        validation_set = [lgbm_train_set, lgbm_eval_set]
        core_params = assoc(core_params, 'valid_sets', validation_set)
    
    
    bst = lgbm.train(hyper_params, lgbm_train_set, **core_params)
    
    def predict(X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        col_dict = {prediction_column: bst.predict(X_test[features].values, num_iteration=bst.best_iteration),
                    real_values_column: y_test}
    
        return X_test.assign(**col_dict)
    
    def measure_metric(df_with_preds, test_metrics):
        import regression_metrics
        dict_metrics = {}
        # double checking for nan
        df_with_preds[real_values_column] = df_with_preds[real_values_column].fillna(df_with_preds[real_values_column].mean())
        df_with_preds[prediction_column] = df_with_preds[prediction_column].fillna(df_with_preds[prediction_column].mean())
        for metric in test_metrics:
            metric_func = getattr(regression_metrics, metric)
            dict_metrics[metric] = metric_func(df_with_preds[real_values_column], df_with_preds[prediction_column])
        return dict_metrics

    df_with_preds = predict(X_test, y_test)
    dict_metrics = measure_metric(df_with_preds, test_metrics)
    
#     if log:
#         from xpml.training import logger
#         if 'task' not in hyper_params:
#             assoc(hyper_params, 'task', 'train')
            
#         log = {'lgbm_regression_model': {
#         'features': features,
#         'target': target,
#         'prediction_column': prediction_column,
#         'package': "lightgbm",
#         'package_version': lgbm.__version__,
#         'parameters': {'hyperparameters': hyper_params, 'core_params': core_params},
#         'feature_importance': dict(zip(features, bst.feature_importance().tolist())),
#         'training_samples': len(X_train),
#         'metrics': dict_metrics},
#         'object': bst}

#         logger.log_regression_tree_model(project_name, log)

    dict_return = {'model_object': bst,
                   'pred_func': predict,
                   'df_with_preds':df_with_preds,
                   'calc_metrics': dict_metrics,
                   'log': log}
    
    return dict_return


def xgb_regressor(train_set: List[pd.DataFrame],
                         test_set: List[pd.DataFrame],
                         features: List[str],
                         target: str,
                         test_metrics: List[str],
                         validation_set: List[pd.DataFrame] = None,
                         core_params: Dict[str, any] = {},
                         hyper_params: Dict[str, any] = {},
                         prediction_column: str = "preds",
                         real_values_column: str = "real_value",
                         log: bool = False,
                         project_name: str = None
                         ) -> Callable:    
    """
    Fits a XGB regressor to the dataset
    
    Parameters
    ----------
    train_set: List of pandas.DataFrame
        [X_train, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.
        
    test_set: List of pandas.DataFrame
        [X_test, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.

    features: list of str
        the list of features used to train the model
        (used for feature importance and log)
    
    target: str
        the target feature used in the model
        (used for log only)
        
    test_metrics: list of str
        list of measurement metrics to use, must be on
        mae: mean absolute error
        mdae: median absolute error
        mse: mean squared error
        rmse: root mean squared error
        msle: mean squared log error
        rmsle: root mean squared error
        r2: r squared
    
    validation_set: list of [train_data, validation_data]
        Lgbm validation set for earling stop
        
    core_params: dict, optional
        core_params of the algorithm
        defaults
        ---------
        num_boost_round, valid_sets=None, valid_names=None,
        fobj=None, feval=None, init_model=None, feature_name='auto',
        categorical_feature='auto', early_stopping_rounds=None,
        evals_result=None, verbose_eval=True, learning_rates=None,
        keep_training_booster=False, callbacks=None
        ---------
        
    hyperparams: dict, optional
        The params of the lgbm regression
        dict in format of {'hyperparameter_name': hyperparameter_value}
        If not passed the default will be used
        
    prediction_column: str
        The name of the column with the predictions from the model
        default `preds`
        
    real_value: str
        The name of the column with the real value from test set
        default `real_value`
        
    log: boll
        Boolean condition if the model log should be generated into mlflow
    
    project_name: str
        Name of the project for log, only used if log = True
 
    """
    
    import xgboost as xgb
    
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    # check mandatory core params
    core_params = core_params if "verbose_eval" in core_params else assoc(core_params, 'verbose_eval', 200)
    core_params = core_params if "num_boost_round" in core_params else assoc(core_params, 'num_boost_round', 1000)

    
    # check mandatory hyper params
    hyper_params = hyper_params if "objective" in hyper_params else assoc(hyper_params, 'objective', 'reg:squarederror')
 
    dtrain = xgb.DMatrix(X_train.values, label = y_train.values)
    
    if validation_set:
        dvalid = xgb.DMatrix(validation_set[0].values, label = validation_set[1].values)
        validation_set = [(dvalid, 'eval')]
        core_params = assoc(core_params, 'evals', validation_set)
        hyper_params = assoc(hyper_params, 'eval_metric', 'mae')
    
    model = xgb.train(params=hyper_params, dtrain=dtrain, **core_params)
    
    def predict(X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        dtest = xgb.DMatrix(X_test.values, label = y_test.values)
        col_dict = {prediction_column: model.predict(dtest),
                    real_values_column: y_test}
    
        return X_test.assign(**col_dict)
    
    
    def measure_metric(df_with_preds, test_metrics):
        import regression_metrics
        dict_metrics = {}
        # double checking for nan
        df_with_preds[real_values_column] = df_with_preds[real_values_column].fillna(df_with_preds[real_values_column].mean())
        df_with_preds[prediction_column] = df_with_preds[prediction_column].fillna(df_with_preds[prediction_column].mean())
        for metric in test_metrics:
            metric_func = getattr(regression_metrics, metric)
            dict_metrics[metric] = metric_func(df_with_preds[real_values_column], df_with_preds[prediction_column])
        return dict_metrics

    df_with_preds = predict(X_test, y_test)
    dict_metrics = measure_metric(df_with_preds, test_metrics)
    
    dict_return = {'model_object': model,
                   'pred_func': predict,
                   'df_with_preds':df_with_preds,
                   'calc_metrics': dict_metrics,
                   'log': log}
        
    
    return dict_return



def random_forest_regressor(train_set: List[pd.DataFrame],
                         test_set: List[pd.DataFrame],
                         features: List[str],
                         target: str,
                         test_metrics: List[str],
#                          validation_set: List[pd.DataFrame] = None,
                         core_params: Dict[str, any] = {},
                         hyper_params: Dict[str, any] = {},
                         prediction_column: str = "preds",
                         real_values_column: str = "real_value",
                         log: bool = False,
                         project_name: str = None
                         ) -> Callable:    
    """
    Fits a XGB regressor to the dataset
    
    Parameters
    ----------
    train_set: List of pandas.DataFrame
        [X_train, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.
        
    test_set: List of pandas.DataFrame
        [X_test, y_test]
        A list of pandas Dataframe with features and target columns. 
        Already transformed.
        The model will be trained to predict the target colum.

    features: list of str
        the list of features used to train the model
        (used for feature importance and log)
    
    target: str
        the target feature used in the model
        (used for log only)
        
    test_metrics: list of str
        list of measurement metrics to use, must be on
        mae: mean absolute error
        mdae: median absolute error
        mse: mean squared error
        rmse: root mean squared error
        msle: mean squared log error
        rmsle: root mean squared error
        r2: r squared
    
    validation_set: list of [train_data, validation_data]
        Lgbm validation set for earling stop
        
    core_params: dict, optional
        core_params of the algorithm
        defaults
        ---------
        num_boost_round, valid_sets=None, valid_names=None,
        fobj=None, feval=None, init_model=None, feature_name='auto',
        categorical_feature='auto', early_stopping_rounds=None,
        evals_result=None, verbose_eval=True, learning_rates=None,
        keep_training_booster=False, callbacks=None
        ---------
        
    hyperparams: dict, optional
        The params of the lgbm regression
        dict in format of {'hyperparameter_name': hyperparameter_value}
        If not passed the default will be used
        
    prediction_column: str
        The name of the column with the predictions from the model
        default `preds`
        
    real_value: str
        The name of the column with the real value from test set
        default `real_value`
        
    log: boll
        Boolean condition if the model log should be generated into mlflow
    
    project_name: str
        Name of the project for log, only used if log = True
 
    """
    
    from sklearn.ensemble import RandomForestRegressor
    
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    # check mandatory core params
    core_params = core_params if "max_depth" in core_params else assoc(core_params, 'max_depth', 3)
    core_params = core_params if "n_estimators" in core_params else assoc(core_params, 'n_estimators', 100)
    core_params = core_params if "random_state" in core_params else assoc(core_params, 'random_state', 42)

    
    model = RandomForestRegressor(**core_params, **hyper_params)
    model.fit(X_train, y_train)
    
    def predict(X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        col_dict = {prediction_column: model.predict(X_test),
                    real_values_column: y_test}
    
        return X_test.assign(**col_dict)
    
    
    def measure_metric(df_with_preds, test_metrics):
        import regression_metrics
        dict_metrics = {}
        # double checking for nan
        df_with_preds[real_values_column] = df_with_preds[real_values_column].fillna(df_with_preds[real_values_column].mean())
        df_with_preds[prediction_column] = df_with_preds[prediction_column].fillna(df_with_preds[prediction_column].mean())
        for metric in test_metrics:
            metric_func = getattr(regression_metrics, metric)
            dict_metrics[metric] = metric_func(df_with_preds[real_values_column], df_with_preds[prediction_column])
        return dict_metrics

    df_with_preds = predict(X_test, y_test)
    dict_metrics = measure_metric(df_with_preds, test_metrics)
    
    dict_return = {'model_object': model,
                   'pred_func': predict,
                   'df_with_preds':df_with_preds,
                   'calc_metrics': dict_metrics,
                   'log': log}
        
    
    return dict_return