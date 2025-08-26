################################################# Some configuration #################################################
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Set this to disable oneDNN optimizations

# Remove existing log file if it exists
log_path = os.path.join("logs", "program.log")
if os.path.exists(log_path): 
    os.remove(log_path)

################################################# Import necessary modules #################################################
from src.utils.utils import load_config_from_yaml
from src.scores.exp_score import calculate_exp_score_nas
import numpy as np
from src.utils.correlation_calculator import get_kendall,get_Spearman
from src.utils.save_files import save_exp_score,save_ranked_accuracies,save_ranked_exp_scores
from src.logging.logger import get_logger
import tensorflow as tf
import random, h5py
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
make_logger = get_logger(__name__)

################################################# Set random seed #################################################
seed_value = 42  # If you don't set the seed, you will get different results every time
random.seed(seed_value)                    
np.random.seed(seed_value)              
tf.random.set_seed(seed_value)

################################################# Load configuration from YAML file #################################################
config_dict = load_config_from_yaml(path="config/config.yaml")
dataset_name = config_dict["dataset"]["name"]
num_models_to_evaluate = config_dict["dataset"]["num_models_to_evaluate"]
input_shape = config_dict["dataset"]["input_shape"]
ranking_method = config_dict["ranking"]["method"]
ranking_type_of_score = config_dict["ranking"]["type_of_score"]

################################################# Get feature maps #################################################
if dataset_name == "cifar10":
    file_path = "fmap_cifar10.h5"
elif dataset_name == "cifar100":
    file_path = "fmap_cifar100.h5"
elif dataset_name == "imagenet":
    file_path = "fmap_imagenet.h5"
else:
    make_logger.error(f"Unknown dataset: {dataset_name} or there is no feature map file.")
    raise ValueError(f"Unknown dataset: {dataset_name}")

# Read fmap dict from h5 file ans store info in famp_dict
fmap_dict = {}
def natural_key(s):
    """
    This function extracts the numeric part from a string formatted as "Layer_X"
    and converts it to an integer for natural sorting.
    """
    # Extract the number after "Layer_" and convert to int
    match = re.search(r'Layer_(\d+)', s, re.IGNORECASE)
    return int(match.group(1)) if match else float('inf')

with h5py.File(file_path, 'r') as f:
    ground_truth_acc_list = []
    model_names_list =[]
    for model_key in f.keys():
        model_group = f[model_key]  # adjust if needed
        # print(f"\nLayers in {model_group.name}:")
        fmap_dict[model_key] = {}
        layer_names = list(model_group.keys())
        layer_names_sorted = sorted(layer_names, key=natural_key)
        fmap_dict[model_key]['layer_names'] = layer_names_sorted
        fmap_dict[model_key]['layer_details'] = [model_group[layer_key].attrs.get('layer_detail', 'N/A') for layer_key in layer_names_sorted]
        fmap_dict[model_key]['fmap_list'] = [model_group[layer_key][:] for layer_key in layer_names_sorted]
        ground_truth_acc_list.append(model_group.attrs.get('test_accuracy', 'N/A'))
        model_names_list.append(model_key)

################################################# Get expressivity_score #################################################
all_expressivity_score_df = calculate_exp_score_nas(fmap_dict = fmap_dict,
                                                 show_exp_score=True)
                                                

################################################# Save expressivity score for each model in specific dataset #################################################
for model_name, expressivity_score_df in all_expressivity_score_df.items():
    save_exp_score(expressivity_score_df, model_name, dataset_name) # This saves the expressivity score for each model in folder results/<dataset_name>/<model_name>

################################################# Save sorted grand_truthaccuracy and expressivity score of all models #################################################
ground_truth_accuracy_df = save_ranked_accuracies(ground_truth_acc_list, model_names_list, dataset_name)
all_expressivity_score_df_ranked = save_ranked_exp_scores(method=ranking_method, score_type=ranking_type_of_score, database_name=dataset_name)

################################################# Get Kendall and Spearman performance #################################################
tau, p_value, merged_df = get_kendall(ground_truth_accuracy_df, all_expressivity_score_df_ranked, method=ranking_method, type_of_score=ranking_type_of_score, database_name=dataset_name)
tau_df = pd.DataFrame({"Kendall's Tau": [tau], "p-value": [p_value]})
tau_df.to_csv(f"results/{dataset_name}/Zero_Cost_Proxy/Kendall_performance.csv", index=False,header=True)

rho, p_value, merged_df = get_Spearman(ground_truth_accuracy_df, all_expressivity_score_df_ranked, method=ranking_method, type_of_score=ranking_type_of_score, database_name=dataset_name)
spearman_df = pd.DataFrame({"Spearman's Rho": [rho], "p-value": [p_value]})
spearman_df.to_csv(f"results/{dataset_name}/Zero_Cost_Proxy/Spearman_performance.csv", index=False,header=True)

if __name__ == "__main__":
    pass
