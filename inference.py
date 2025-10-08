################################################# Set some configuration #################################################
import os
import warnings
import logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING logs from TF C++ backend
logging.getLogger('tensorflow').setLevel(logging.ERROR)

log_path = os.path.join("logs", "program.log")
if os.path.exists(log_path): 
    os.remove(log_path) #Remove existing log file if it exists

################################################# Import necessary modules #################################################
from src.utils.utils import load_config_from_yaml
from src.scores.exp_score import calculate_exp_score_nas
import numpy as np
from src.utils.save_files import save_exp_score
from src.utils.get_feature_maps import extract_feature_maps_from_torch,extract_feature_maps_from_keras
import tensorflow as tf
import torch,warnings
from src.logging.logger import get_logger
make_logger = get_logger(__name__) # Create Logger

################################################# Load configuration from YAML file #################################################
config_dict = load_config_from_yaml(path="config/config.yaml")
dataset_name = config_dict["dataset"]["name"]
num_models_to_evaluate = config_dict["dataset"]["num_models_to_evaluate"]
input_shape = config_dict["dataset"]["input_shape"]
method = config_dict["ranking"]["method"]
ranking_type_of_score = config_dict["ranking"]["type_of_score"]
batch_size = config_dict["dataset"]["batch_size"]
run_version = config_dict["run_conf"]["run_version"]
model_type = config_dict['model_type']
trim_mean_value = config_dict["run_conf"]["trim_mean_value"]

################################################# Set Seed Value #################################################
seed_value = 42  # Set this to see same result  
np.random.seed(seed_value)              
tf.random.set_seed(seed_value)
torch.manual_seed(seed_value)

################################################# Build Model #################################################
# Change this for building your model, if needed.
if model_type=="keras":
    from src.models.keras_models import build_model
    our_model = build_model() 
elif model_type=="torch":
    from src.models.torch_models import SimpleCIFAR100Model_torch
    our_model =  SimpleCIFAR100Model_torch()


################################################# Run Program #################################################
if model_type=="keras":
    try: 
        h,w,c =input_shape[0],input_shape[1],input_shape[2]     # Read input shape
        keras_random_input = torch.randn(batch_size, h, w, c).numpy() # Generate random data
    
    except:
        h,c = input_shape[0],input_shape[1] # Read input shape 
        keras_random_input = torch.randn(batch_size, h, c).numpy()
    
    try:
        keras_model = our_model # Read Model
    except :
        print('There is an error in reading model. Your model must have some specific structure')
    
    keras_model.summary()  # Print Summary
    print('Wait for extracting feature maps...')
    fmap_dict = extract_feature_maps_from_keras(model=keras_model,x=keras_random_input) # Extract fmap
    print('Wait for calculating scores...')
    expressivity_score_df = calculate_exp_score_nas(fmap_dict = fmap_dict,
                                                show_exp_score=False,method=method,trim_mean_value = trim_mean_value)  # Get Expressivity score                             
    for model_name, expressivity_score_df in expressivity_score_df.items():
        final_expressivity_score = expressivity_score_df[expressivity_score_df['Layer Name']==method]['Normalized Expressivity Score']
        save_exp_score(expressivity_score_df, model_name, dataset_name,run_version) # This saves the expressivity score for each model in folder results/<dataset_name>/<model_name>
        print('*'*100)
        print('*'*100)
        print(f'You can see the result in directory: ./results/{dataset_name}/{run_version}')
        make_logger.info(f"Final expressivity score:{float(final_expressivity_score)} ",)
    
elif model_type=='torch':
    try:
        c,h,w =input_shape[0],input_shape[1],input_shape[2]
        torch_random_input = torch.randn(batch_size, c,h, w)
    
    except:
        c,h = input_shape[0],input_shape[1]
        torch_random_input = torch.randn(batch_size, c, h)
    
    try:
        torch_model = our_model
    except :
        print('There is an error in reading model. Your model must have some specific structure')
    
    fmap_dict = extract_feature_maps_from_torch(model=torch_model,x=torch_random_input)
    expressivity_score_df = calculate_exp_score_nas(fmap_dict = fmap_dict,
                                                show_exp_score=False,method=method,trim_mean_value = trim_mean_value)                                ################################################# Get feature maps #################################################
    for model_name, expressivity_score_df in expressivity_score_df.items():
        final_expressivity_score = expressivity_score_df[expressivity_score_df['Layer Name']==method]['Normalized Expressivity Score']
        save_exp_score(expressivity_score_df, model_name, dataset_name,run_version) # This saves the expressivity score for each model in folder results/<dataset_name>/<model_name>
        print('*'*100)
        print('*'*100)
        print(f'You can see the result in directory: ./results/{dataset_name}/{run_version}')
        make_logger.info(f"Final expressivity score:{float(final_expressivity_score)} ",)

else:
    print('You have chosen the wrong model type.Please choose keras or torch') 

if __name__ == "__main__":
    pass




