import yaml
from src.logging.logger import get_logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

def plot_correlation_performamce(predicted_ranking, ground_truth_ranking, save_path):
    """
    This function plots the correlation performance and save it.
    Args:
        predicted_ranking: the ranking predicted by zero-cost proxy method
        ground_truth_ranking: the ranking of models got from NATS Benchmark,
        save_path: the path we want to save this plot
        
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    x = predicted_ranking
    y = ground_truth_ranking
    ax = plt.scatter(x, y, s=50)  # Increase marker size with 's' parameter
    
    # Randomly select 10% of data points to annotate
    num_points = len(x)
    num_to_annotate = max(1, int(0.2 * num_points))  # Ensure at least one point is annotated
    indices_to_annotate = np.random.choice(num_points, num_to_annotate, replace=False)
    
    for i in indices_to_annotate:
        plt.text(x[i], y[i], f"({int(x[i])}, {int(y[i])})", fontsize=16, ha='center')
        plt.scatter(x[i], y[i], color='green', s=70)  # Change marker to red and slightly increase size
    
    plt.title("Correlation Performance", fontsize=16, fontweight='bold')
    plt.xlabel("Predicted Ranking by Our Work", fontsize=14, fontweight='bold')
    plt.ylabel("Ground Truth Ranking", fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.savefig(save_path, dpi=900)
    plt.close(fig)