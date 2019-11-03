import os
from configparser import ConfigParser
from urllib.request import urlretrieve
import zipfile

from config import read_conf

def download_and_extract_zip(file_link: str, path: str, filename:str) -> None:
    """
    Download and extract zip file
    """
    download_path = f'{path}/{filename}.zip'
    # download file
    filehandle, _ = urlretrieve(file_link, download_path)
    # unzip file
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(f'{path}/')

    print (f"{filename} downloaded")
    return None
    
    
def get_ml_challenge_data(config_parser: ConfigParser()) -> None:
    """
    Download data from AutoML challenge
    http://automl.chalearn.org/data
    """
    all_links = dict(config_parser.items('ML_CHALLENGE_DATA_LINKS'))
    ml_challenge_data_path = dict(config_parser.items('ML_CHALLENGE_DATA_PATH'))['path']
    ml_challenge_data_sets = []
    # itera pelas configuracoes baixando os arquivos se necessario
    for file in all_links:
        # checa se o diretorio do dados ja existe
        path_to_check = f"{ml_challenge_data_path}/{file}"
        if not (os.path.isdir(path_to_check)):
            # cria diretorio
            os.makedirs(path_to_check)
            # baixa e extrai arquivo
            download_and_extract_zip(all_links[file], path_to_check, file)
        ml_challenge_data_sets.append(file)
    print("All ML_CHALLENGE files ready!")
    return ml_challenge_data_sets


def get_all_data(only: str = None):
    """
    Download all data necessary for this project
    """
    config_parser = read_conf.read_conf_file()
    all_ready_datasets = []
    # ml challenge data
    ml_challenge_data_sets = get_ml_challenge_data(config_parser)
    
    if only == 'ml_challenge':
        return ml_challenge_data_sets
    
    all_ready_datasets = all_ready_datasets = ml_challenge_data_sets
    return all_ready_datasets
    #kaggle data...