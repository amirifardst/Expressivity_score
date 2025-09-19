
import pandas as pd
import numpy as np
from src.logging.logger import get_logger
make_logger = get_logger(__name__)
from scipy.stats import trim_mean

def get_statistics(exp_score_dict, show_exp_score=True, method="trim_mean", trim_mean_value= 0.4):
    """
    This function calculates statistics from the expressivity score dictionary for each model.
    Args:
        exp_score_dict: A Dictionary containing expressivity scores for each layer of the model.
        show_exp_score: A Boolean to control whether to print the DataFrame.
    Returns:
        exp_score_df: Expressivity DataFrame with statistics.
    """

    make_logger.info('Calculating statistics from expressivity score dictionary started')
    
    # Create DataFrame from the expressivity score dictionary
    exp_score_df = pd.DataFrame({
        'Layer Name': list(exp_score_dict.keys()),
        "Detail": [v['layer_detail'] for v in exp_score_dict.values()],
        'Spatial Size': [int(v['spatial_size']) for v in exp_score_dict.values()],
        'Number of Channels': [int(v['num_channels']) for v in exp_score_dict.values()],
        "Log (c)": [float(v['log_c']) for v in exp_score_dict.values()],
        'Expressivity Score': [float(v['expressivity_score']) for v in exp_score_dict.values()],
        'Normalized Expressivity Score': [float(v['normalized_expressivity_score']) for v in exp_score_dict.values()],
    })



    
    min_row = pd.DataFrame({
        'Layer Name': ['min'],
        'Detail': [''],
        'Spatial Size': np.min([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.min([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.min([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.min([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.min([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })



    #add max value to the DataFrame
    max_row = pd.DataFrame({
        'Layer Name': ['max'],
        'Detail': [''],
        'Spatial Size': np.max([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.max([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.max([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.max([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.max([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })
    # Add trim_mean
    if method == "trim_mean":
        trim_mean_score = trim_mean([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()], trim_mean_value) 
        assert isinstance(trim_mean_score,float)
        avg_row = pd.DataFrame({
            'Layer Name': ['trim_mean'],
            'Detail': [''],
            'Spatial Size': np.mean([int(v['spatial_size']) for v in exp_score_dict.values()]),
            'Number of Channels': np.mean([int(v['num_channels']) for v in exp_score_dict.values()]),
            'Log (c)': np.mean([float(v['log_c']) for v in exp_score_dict.values()]),
            'Expressivity Score': trim_mean([float(v['expressivity_score']) for v in exp_score_dict.values()], trim_mean_value),
            "Normalized Expressivity Score":trim_mean_score ,
            })
    elif method == "mean":
            mean_score = np.mean([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()])
            assert isinstance(mean_score,float)
            avg_row = pd.DataFrame({
            'Layer Name': ['mean'],
            'Detail': [''],
            'Spatial Size': np.mean([int(v['spatial_size']) for v in exp_score_dict.values()]),
            'Number of Channels': np.mean([int(v['num_channels']) for v in exp_score_dict.values()]),
            'Log (c)': np.mean([float(v['log_c']) for v in exp_score_dict.values()]),
            'Expressivity Score': np.mean([float(v['expressivity_score']) for v in exp_score_dict.values()]),
            "Normalized Expressivity Score":mean_score ,
            })  

    # add median
    median_row = pd.DataFrame({
        'Layer Name': ['median'],
        'Detail': [''],
        'Spatial Size': np.median([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.median([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.median([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.median([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.median([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })
    # Add std
    std_row = pd.DataFrame({
        'Layer Name': ['std'],
        'Detail': [''],
        'Spatial Size': np.std([int(v['spatial_size']) for v in exp_score_dict.values()]),
        'Number of Channels': np.std([int(v['num_channels']) for v in exp_score_dict.values()]),
        'Log (c)': np.std([float(v['log_c']) for v in exp_score_dict.values()]),
        'Expressivity Score': np.std([float(v['expressivity_score']) for v in exp_score_dict.values()]),
        "Normalized Expressivity Score": np.std([float(v['normalized_expressivity_score']) for v in exp_score_dict.values()]),
    })  
    exp_score_df = pd.concat([exp_score_df, min_row, max_row, avg_row, median_row, std_row], ignore_index=True)

    if show_exp_score:
        print(exp_score_df)

    return exp_score_df