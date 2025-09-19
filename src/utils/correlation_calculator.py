import pandas as pd
from scipy.stats import kendalltau,spearmanr
from src.logging.logger import get_logger
from src.utils.utils import plot_correlation_performamce
make_logger = get_logger(__name__)


# Load the CSV files
def get_kendall_for_exp(Grand_truth_accuracy_df, expressivity_score_df, method="mean", type_of_score="Normalized Expressivity Score", database_name="cifar10", run_version="v_01"):
    """
    Calculate Kendall's Tau correlation between model accuracy and expressivity scores .
    Args:
        Grand_truth_accuracy_df (str): Path to the CSV file containing model accuracies.
        expressivity_score_df (str): Path to the CSV file containing expressivity scores.
        method (str): The method used for ranking (e.g., "mean").
        type_of_score (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        tau (float): Kendall's Tau correlation coefficient.
        p_value (float): Two-tailed p-value for the correlation.
        merged_df (DataFrame): DataFrame containing the merged data with ranks.
    """
    make_logger.info("Getting kendall correlation started...")
    #Create a Merged_df on model_name "Model" to align models present in both files
    merged_df = pd.merge(Grand_truth_accuracy_df, expressivity_score_df, on='Model')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.
    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min').astype(int)
    merged_df['rank_exp_score'] = merged_df[f'{method}_{type_of_score}'].rank(ascending=False, method='first').astype(int)

    # Compute Kendall's Tau correlation between the two ranks
    tau, p_value = kendalltau(merged_df['rank_accuracy'], merged_df['rank_exp_score'])

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(f'results/{database_name}/{run_version}/Zero_Cost_Proxy/Expressivity_correlation.csv', index=False)
    make_logger.info(f"Kendall's Tau {tau} with p-value {p_value} correlation calculated and saved to results/{database_name}/Zero_Cost_Proxy/Expressivity_kendall.csv")
    print(f"Kendall's Tau correlation: {tau}, p-value: {p_value}")
    plot_correlation_performamce(predicted_ranking = merged_df['rank_accuracy'] , ground_truth_ranking=merged_df['rank_exp_score'],save_path=f'results/{database_name}/{run_version}/Zero_Cost_Proxy/Expressivity_correlation.png')
    
    return tau, p_value, merged_df

def get_kendall_for_prog(Grand_truth_accuracy_df, progssivity_score_df, method="min", type_of_score="Progressivity_score", database_name="cifar10", run_version="v_01"):
    """
    Calculate Kendall's Tau correlation between model accuracy and expressivity scores .
    Args:
        Grand_truth_accuracy_df (str): Path to the CSV file containing model accuracies.
        expressivity_score_df (str): Path to the CSV file containing expressivity scores.
        method (str): The method used for ranking (e.g., "mean").
        type_of_score (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        tau (float): Kendall's Tau correlation coefficient.
        p_value (float): Two-tailed p-value for the correlation.
        merged_df (DataFrame): DataFrame containing the merged data with ranks.
    """
    make_logger.info("Getting kendall correlation started...")
    #Create a Merged_df on model_name "Model" to align models present in both files
    merged_df = pd.merge(Grand_truth_accuracy_df, progssivity_score_df, on='Model')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.
    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min').astype(int)
    merged_df['rank_prog_score'] = merged_df[f'{method}_{type_of_score}'].rank(ascending=True, method='first').astype(int)

    # Compute Kendall's Tau correlation between the two ranks
    tau, p_value = kendalltau(merged_df['rank_accuracy'], merged_df['rank_prog_score'])

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(f'results/{database_name}/{run_version}/Zero_Cost_Proxy/Progssivity_correlation.csv', index=False)
    make_logger.info(f"Kendall's Tau {tau} with p-value {p_value} correlation calculated and saved to results/{database_name}/Zero_Cost_Proxy/Progssivity_kendall.csv")
    print(f"Kendall's Tau correlation: {tau}, p-value: {p_value}")
    plot_correlation_performamce(predicted_ranking = merged_df['rank_accuracy'] , ground_truth_ranking=merged_df['rank_prog_score'],save_path=f'results/{database_name}/{run_version}/Zero_Cost_Proxy/Progssivity_correlation.png')
    
    return tau, p_value, merged_df



def get_Spearman_for_exp(Grand_truth_accuracy_df, expressivity_score_df, method="mean", type_of_score="Normalized Expressivity Score", database_name="cifar10", run_version="v_01"):
    """
    This function uses Spearman's rank correlation calculation.
    Args:
        Grand_truth_accuracy_df (str): Path to the CSV file containing model accuracies.
        expressivity_score_df (str): Path to the CSV file containing expressivity scores.
        method (str): The method used for ranking (e.g., "mean").
        type_of_score (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        rho: Spearman's Rho correlation coefficient.
        p_value: p-value for the correlation.
        merged_df: DataFrame containing the merged data with ranks.
    """
    make_logger.info("Getting spearman correlation started...")
    #Create a Merged_df on model_name "Model" to align models present in both files
    merged_df = pd.merge(Grand_truth_accuracy_df, expressivity_score_df, on='Model')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.
    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min')
    merged_df['rank_exp_score'] = merged_df[f'{method}_{type_of_score}'].rank(ascending=False, method='min')

    # Compute Kendall's Tau correlation between the two ranks
    rho, p_value = spearmanr(merged_df['rank_accuracy'], merged_df['rank_exp_score'])

    make_logger.info(f"Spearman's Rho {rho} with p-value {p_value} correlation calculated.")
    print(f"Spearman's Rho correlation: {rho}, p-value: {p_value}")
    return rho, p_value, merged_df

def get_Spearman_for_prog(Grand_truth_accuracy_df, progssivity_score_df, method="min", type_of_score="Progressivity_score", database_name="cifar10", run_version="v_01"):

    """
    This function uses Spearman's rank correlation calculation.
    Args:
        Grand_truth_accuracy_df (str): Path to the CSV file containing model accuracies.
        expressivity_score_df (str): Path to the CSV file containing expressivity scores.
        method (str): The method used for ranking (e.g., "mean").
        type_of_score (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        rho: Spearman's Rho correlation coefficient.
        p_value: p-value for the correlation.
        merged_df: DataFrame containing the merged data with ranks.
    """
    make_logger.info("Getting spearman correlation started...")
    #Create a Merged_df on model_name "Model" to align models present in both files
    merged_df = pd.merge(Grand_truth_accuracy_df, progssivity_score_df, on='Model')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.
    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min')
    merged_df['rank_prog_score'] = merged_df[f'{method}_{type_of_score}'].rank(ascending=True, method='min')

    # Compute Kendall's Tau correlation between the two ranks
    rho, p_value = spearmanr(merged_df['rank_accuracy'], merged_df['rank_prog_score'])

    make_logger.info(f"Spearman's Rho {rho} with p-value {p_value} correlation calculated.")
    print(f"Spearman's Rho correlation: {rho}, p-value: {p_value}")
    return rho, p_value, merged_df


def get_kendall_for_both(Grand_truth_accuracy_df, aggregated_df, database_name="cifar10", run_version="v_01"):
    """
    Calculate Kendall's Tau correlation between model accuracy and expressivity scores .
    Args:
        Grand_truth_accuracy_df (str): Path to the CSV file containing model accuracies.
        expressivity_score_df (str): Path to the CSV file containing expressivity scores.
        method (str): The method used for ranking (e.g., "mean").
        type_of_score (str): The type of score used for ranking (e.g., "Normalized Expressivity Score").
        database_name (str): The name of the database.
    Returns:
        tau (float): Kendall's Tau correlation coefficient.
        p_value (float): Two-tailed p-value for the correlation.
        merged_df (DataFrame): DataFrame containing the merged data with ranks.
    """
    make_logger.info("Getting kendall correlation started...")
    #Create a Merged_df on model_name "Model" to align models present in both files
    merged_df = pd.merge(Grand_truth_accuracy_df, aggregated_df, on='Model')

    # Since the CSVs are sorted by accuracy and exp_score respectively,
    # we need to assign ranks based on their order in each CSV.
    # Assign ranks based on order in each CSV (starting from 1)
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min').astype(int)

    # Compute Kendall's Tau correlation between the two ranks
    tau, p_value = kendalltau(merged_df['rank_accuracy'], merged_df['Rank_by_both'])

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(f'results/{database_name}/{run_version}/Zero_Cost_Proxy/aggregated_correlation.csv', index=False)
    make_logger.info(f"Aggregated Kendall's Tau {tau} with p-value {p_value} correlation calculated and saved to results/{database_name}/Zero_Cost_Proxy/Expressivity_kendall.csv")
    print(f"Aggregated Kendall's Tau correlation: {tau}, p-value: {p_value}")
    plot_correlation_performamce(predicted_ranking = merged_df['rank_accuracy'] , ground_truth_ranking=merged_df['Rank_by_both'],save_path=f'results/{database_name}/{run_version}/Zero_Cost_Proxy/aggregated_correlation.png')
    
    return tau, p_value, merged_df
