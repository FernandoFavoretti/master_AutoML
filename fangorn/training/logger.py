def set_server():
    import mlflow
    import os
    server = os.environ['MLFLOW_TRACKING_URI']
    mlflow.tracking.set_tracking_uri(server)
    return None

def get_experiment(name):
    import mlflow
    set_server()
    all_experiments = mlflow.tracking.list_experiments()
    for experiment in all_experiments:
        if experiment.__dict__['_name'] == name:
            return experiment.__dict__['_experiment_id']
    
    exp_id = create_new_experiment(name)
    print(" experiment {} - id {} created.".format(name, exp_id))
    return exp_id

def create_new_experiment(name):
    import mlflow
    set_server()
    return mlflow.tracking.create_experiment(name)

def log_params_dict(exp_name, params):
    import mlflow
    for param,value in params.items():
        mlflow.tracking.log_param(param, value)

def log_metric_dict(exp_name, metrics):
    import mlflow
    for metric,value in metrics.items():
        mlflow.tracking.log_metric(metric, value)
    
def log_artifact(exp_name, file_path):
    import mlflow
    import os
    mlflow.tracking.log_artifact(file_path)
    os.remove(file_path)
    
def log_experiment(exp_name, metrics, hyper_params, core_params, fig_name=None, model=None):
    import mlflow
    import os
    set_server()
    exp_id = get_experiment(exp_name)
    with mlflow.tracking.start_run(experiment_id=exp_id):
        log_params_dict(exp_name, {**hyper_params,**core_params})
        log_metric_dict(exp_name, metrics)
        if fig_name:
            log_artifact(exp_name, fig_name)
        if model:
            log_artifact(exp_name, model)
    mlflow.tracking.end_run()
        
        
def log_regression_tree_model(exp_name, log):
    exp_id = get_experiment(exp_name)
    def log_feature_importance():
        # plot and save feature importance fig fig as artifact
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        plt.ioff()
        set_server()
        fig_name = "feature_importance_{}.png".format(exp_name)
        
        feature_importance = log['lgbm_regression_model']['feature_importance']
        df_importance = pd.DataFrame(feature_importance, index=[0]).T
        df_importance = df_importance.sort_values(by=[0],ascending=False)
        feature_importance = df_importance.plot(kind='barh',
                           figsize=(df_importance.shape[0]/2,df_importance.shape[0]/2),
                          fontsize = df_importance.shape[0]/2)
        
        fig = feature_importance.get_figure()
        fig.savefig(fig_name)
        fig.clf()
        return fig_name
    
    def save_model():
        import joblib
        import pandas as pd
        model = log['object']
        filename = 'modelo_{}.xpml'.format(exp_name)
        joblib.dump(model, filename)
        return filename
    
    def prepare_metrics_and_params(exp_name, log):
        hyper_params = log['lgbm_regression_model']['parameters']['hyperparameters']
        core_params = log['lgbm_regression_model']['parameters']['core_params']
        metrics = log['lgbm_regression_model']['metrics']
        return hyper_params, core_params, metrics
    
    hyper_params, core_params, metrics = prepare_metrics_and_params(exp_name, log)
    fig_name = log_feature_importance()
    model_file = save_model()
    log_experiment(exp_name, metrics, hyper_params, core_params, fig_name, model_file)