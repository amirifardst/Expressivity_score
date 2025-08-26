import datetime
import os
import pandas as pd
from src.logging.logger import get_logger
from IPython.display import display
from datetime import datetime
import numpy as np


make_logger = get_logger(__name__)

def save_exp_score(exp_score_df, model_name, database_name):
    """
    This function saves the expressivity score DataFrame of each model to a CSV file located in the results directory/<database_name>/<model_name>.
    Args:
        exp_score_df (DataFrame): The expressivity score DataFrame to save.
        model_name (str): The name of the model.
        database_name (str): The name of the database.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results/{database_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    exp_score_df.to_csv(f"{save_dir}/{model_name}_{timestamp}_score.csv", mode="a", header=True, index=True)
    make_logger.info(f"Expressivity scores of model {model_name} saved successfully.")

def save_ranked_accuracies(accuracy_list, model_names_list, database_name):
    """
    This function saves all model accuracies to a csv file in ranked order.
    Args:
        accuracy_list (list): The list of accuracies of the models.
        model_names_list (list): The list of model names.
        database_name (str): The name of the database.
    Returns:
        accuracy_df (DataFrame): DataFrame containing model names and their accuracies.

    """
    sorted_acc_list = sorted(zip(model_names_list, accuracy_list), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(sorted_acc_list, columns=["Model", "Accuracy"])
    save_dir = f"results/{database_name}/Zero_Cost_Proxy"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/Grand_Truth_Accuracy.csv", mode="a", header=True, index=True)
    make_logger.info(f"Ranked accuracies of all models saved successfully in {save_dir}/Grand_Truth_Accuracy.csv")
    return df

def save_ranked_exp_scores(method="mean", score_type="Normalized Expressivity Score", database_name="cifar10"):
    """
    This function saves the ranked expressivity scores DataFrame to a CSV file.
    Args:
        method (str): The method used for ranking (e.g., "mean").
        score_type (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        scores_df (DataFrame): DataFrame containing the ranked expressivity scores.
    """

    # Path to the results directory
    results_dir = f'results/{database_name}'
    nas_dir = os.path.join(results_dir, 'Zero_Cost_Proxy')
    os.makedirs(nas_dir, exist_ok=True)

    scores = []

    # Loop through folders in results directory
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('model'):
            # Find the first CSV file in the folder
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not csv_files:
                continue
            csv_path = os.path.join(folder_path, csv_files[0])
            df = pd.read_csv(csv_path, index_col=0)
            # Find the row where 'Layer Name' column is 'method'
            row = df[df['Layer Name'] == method]
            if not row.empty:
                value = row[f'{score_type}'].values[0]
                scores.append({'Model': folder, f'{method}_{score_type}': value})

    # Save to DataFrame and CSV
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values(by=f'{method}_{score_type}', ascending=False, ignore_index=True)
    output_path = os.path.join(nas_dir, 'Ranked_Expressivity_Scores.csv')
    scores_df.to_csv(output_path, index=False)
    make_logger.info(f"Ranked expressivity scores saved to {output_path}")

    return scores_df

# def save_accuracy(model_name, database_name, accuracy, val_accuracy):
#     """
#     This function saves the model accuracy to a text file.
#     Args:
#         accuracy (float): The accuracy of the model.
#         model_name (str): The name of the model.
#     """
#     df = pd.DataFrame({"Model Name": [model_name], "Accuracy": [accuracy], "Validation Accuracy": [val_accuracy]})
#     save_dir = f"results/{database_name}/{model_name}"
#     os.makedirs(save_dir, exist_ok=True)
#     df.to_csv(f"{save_dir}/{model_name}_accuracy.csv", mode="a", header=True, index=True)
#     make_logger.info(f"Accuracy of model {model_name} saved successfully.")

# def save_model(model, model_name, database_name):
#     # Create the directory if it doesn't exist
#     save_dir = f"results/{database_name}/{model_name}"
#     os.makedirs(save_dir, exist_ok=True)
#     model.save(os.path.join(save_dir, f'{model_name}.h5'))
#     make_logger.info(f"Model {model_name} saved to {save_dir}")




