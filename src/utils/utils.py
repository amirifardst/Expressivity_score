import yaml
from src.logging.logger import get_logger
make_logger = get_logger(__name__)

def load_config_from_yaml(path):
    """
    Load configuration from a YAML file.
    Args:
        path (str): Path to the YAML configuration file.
    Returns:
        config_dict: Loaded configuration as a dictionary.
    """

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict

