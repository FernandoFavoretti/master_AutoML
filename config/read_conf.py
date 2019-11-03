from pathlib import Path
from configparser import ConfigParser

def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def read_conf_file():
    path = get_project_root()
    config_parser = ConfigParser()
    config_parser.read(f"{path}/config/data_links.ini")
    return config_parser