import pandas as pd
from scipy.stats import kendalltau
from src.logging.logger import get_logger
make_logger = get_logger(__name__)


# Load the CSV files
def get_kendall(Grand_truth_accuracy_df, expressivity_score_df, method="mean", type_of_score="Normalized Expressivity Score", database_name="cifar10"):
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
    merged_df['rank_accuracy'] = merged_df['Accuracy'].rank(ascending=False, method='min')
    merged_df['rank_exp_score'] = merged_df[f'{method}_{type_of_score}'].rank(ascending=False, method='min')

    # Compute Kendall's Tau correlation between the two ranks
    tau, p_value = kendalltau(merged_df['rank_accuracy'], merged_df['rank_exp_score'])

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(f'results/{database_name}/Zero_Cost_Proxy/kendall.csv', index=False)
    make_logger.info(f"Kendall's Tau {tau} with p-value {p_value} correlation calculated and saved to results/{database_name}/Zero_Cost_Proxy/kendall.csv")
    print(f"Kendall's Tau correlation: {tau}, p-value: {p_value}")
    return tau, p_value, merged_df
