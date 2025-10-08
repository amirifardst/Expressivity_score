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
from src.utils.correlation_calculator import get_kendall_for_exp,get_Spearman_for_exp,get_kendall_for_prog,get_Spearman_for_prog,get_kendall_for_both
from src.utils.save_files import save_exp_score,save_ranked_accuracies,save_ranked_exp_scores,save_ranked_prog_scores
from src.logging.logger import get_logger
from src.utils.get_feature_maps import connect_api, extract_feature_maps_evaluation,show_model_summary
import tensorflow as tf
import random, shutil,torch,warnings
import pandas as pd
import time

warnings.filterwarnings("ignore", category=RuntimeWarning)
make_logger = get_logger(__name__)

################################################# Set random seed #################################################
start = time.time()
seed_value = 42  # If you don't set the seed, you will get different results every time
random.seed(seed_value)                    
np.random.seed(seed_value)              
tf.random.set_seed(seed_value)

################################################# Load configuration from YAML file #################################################
config_dict = load_config_from_yaml(path="config/config.yaml")
dataset_name = config_dict["dataset"]["name"]
num_models_to_evaluate = config_dict["dataset"]["num_models_to_evaluate"]
input_shape = config_dict["dataset"]["input_shape"]
method = config_dict["ranking"]["method"]
ranking_type_of_score = config_dict["ranking"]["type_of_score"]
batch_size = config_dict["dataset"]["batch_size"]
run_version = config_dict["run_conf"]["run_version"]
trim_mean_value = config_dict["run_conf"]["trim_mean_value"]

################################################# Download dataset and connect to API: #################################################
API = connect_api()

################################################# Create Random Input #################################################
torch.manual_seed(42)
random_input = torch.randn(batch_size, *input_shape)

################################################# Get feature maps #################################################
model_names_list, ground_truth_acc_list, fmap_dict = extract_feature_maps_evaluation(API, num_models_to_evaluate, dataset_name, random_input)


################################################# Get expressivity_score #################################################
all_expressivity_score_df = calculate_exp_score_nas(fmap_dict = fmap_dict,
                                                 show_exp_score=False,dataset_name=dataset_name,method=method,trim_mean_value = trim_mean_value)
                                                
################################################# Save expressivity score for each model in specific dataset #################################################
for model_name, expressivity_score_df in all_expressivity_score_df.items():
    save_exp_score(expressivity_score_df, model_name, dataset_name,run_version) # This saves the expressivity score for each model in folder results/<dataset_name>/<model_name>

################################################# Save sorted grand_truthaccuracy and expressivity  score of all models #################################################
ground_truth_accuracy_df = save_ranked_accuracies(ground_truth_acc_list, model_names_list, dataset_name,run_version)
all_expressivity_score_df_ranked = save_ranked_exp_scores(method=method, score_type=ranking_type_of_score, database_name=dataset_name,run_version=run_version)

################################################# Get Kendall and Spearman performance #################################################
tau, p_value, merged_df = get_kendall_for_exp(ground_truth_accuracy_df, all_expressivity_score_df_ranked, method=method, type_of_score=ranking_type_of_score, database_name=dataset_name,run_version= run_version)
tau_df = pd.DataFrame({"Kendall's Tau": [tau], "p-value": [p_value]})
tau_df.to_csv(f"results/{dataset_name}/{run_version}/Zero_Cost_Proxy/Expressivity_Kendall_performance.csv", index=False,header=True)


rho, p_value, merged_df = get_Spearman_for_exp(ground_truth_accuracy_df, all_expressivity_score_df_ranked, method=method, type_of_score=ranking_type_of_score, database_name=dataset_name)
spearman_df = pd.DataFrame({"Spearman's Rho": [rho], "p-value": [p_value]})
spearman_df.to_csv(f"results/{dataset_name}/{run_version}/Zero_Cost_Proxy/Expressivity_Spearman_performance.csv", index=False,header=True)

shutil.copy("config/config.yaml",f"results/{dataset_name}/{run_version}") # copy config to the direcotry.

stop = time.time()
################################################# Get Run time #################################################

runtime = int((stop-start))
runtime_df = pd.DataFrame({"runtime(Seconds)":[runtime]})
runtime_df.to_csv(f"results/{dataset_name}/{run_version}/Zero_Cost_Proxy/runtime.csv",index=False,header=True)
print("*"*100)
print(f'You can see the main results in directory ./results/{dataset_name}/{run_version}/Zero_Cost_Proxy')
if __name__ == "__main__":
    pass
