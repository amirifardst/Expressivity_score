import tensorflow as tf
from src.logging.logger import get_logger
import pandas as pd
import numpy as np
import os
from src.utils.statistics import get_statistics
make_logger = get_logger(__name__)


def calculate_exp_score_nas(fmap_dict, show_exp_score=True):
    """
    This function calculates the expressivity score of each model.

    Args: 
        fmap_dict: feature map dictionary, including model/models feature maps
        show_exp_score : Whether to display the expressivity score

    Returns:
        all_exp_score_dict: A dictionary containing the expressivity scores and some statistics for each architecture

    """


    num_architectures = len(fmap_dict.keys())
    architectures = list(fmap_dict.keys())
    make_logger.info("=*"*20)
    make_logger.info(f"Calculating expressivity score has been started")
    make_logger.info(f"Number of architectures: {num_architectures}")
    make_logger.info("=*"*20)
    
    all_exp_score_dict = {}

    # Loop through architectures and save their expressivity scores
    for i in range(num_architectures):
        make_logger.info(f"Processing  {architectures[i]}")
        make_logger.info('-'*50)
        exp_score_dict = {}

        # Loop through feature maps of the current architecture
        fmap_list = fmap_dict[architectures[i]]['fmap_list']
        j = 0
        for fmap in fmap_list:
            try:
                layer_name = fmap_dict[architectures[i]]['layer_names'][j]
                layer_detail = fmap_dict[architectures[i]]['layer_details'][j]
                # make_logger.info(f"Processing layer_{j} named: {layer_name} with detail: {layer_detail}")

                # check if fmap is a TensorFlow tensor
                if not isinstance(fmap, tf.Tensor):
                    fmap = tf.convert_to_tensor(fmap, dtype=tf.float32)

                # Start Calculation
                # Step1: Reshaping fmap to 2d dimensions
                if fmap.ndim == 4: # If fmap is 4D
                    b, c, w, h = fmap.shape # b=batch_size, c=num_channels, w=width, h=height
                    spatial_size = w * h
                    n = b * w * h
                    fmap_2d = tf.reshape(fmap, (n, c))
                    # make_logger.info(f'For this layer--> [b,c,w,h]: [{b},{c},{w},{h}], spatial size: {spatial_size}, fmap_2d shape: {fmap_2d.shape}')

                
                elif fmap.ndim == 2:  # If fmap is already 2D, assume shape is (n, c)
                    n, c = fmap.shape
                    spatial_size = 1 
                    fmap_2d = fmap
                    # make_logger.info(f'For this layer--> [b,c,w,h]: [{b},{c},{w},{h}], spatial size: {spatial_size}, fmap_2d shape: {fmap_2d.shape}')
                
                else:
                    continue  # Skip if not 2D or 4D


                # Step 2: Center the 2d fmap
                mean = tf.reduce_mean(fmap_2d, axis=0, keepdims=True) # Mean over rows (dim=0), keep dims for broadcasting
                fmap_centered = fmap_2d - mean  # shape (n, c)
                # make_logger.info(f'For this layer--> Mean.shape: {mean.shape}, fmap_centered shape: {fmap_centered.shape}')


                # Step 3: Calculate covariance matrix
                Covariance_matrix = tf.matmul(fmap_centered, fmap_centered, transpose_a=True) / tf.cast(tf.shape(fmap_centered)[0], tf.float32) # Covariance matrix: (c, c)
                # make_logger.info(f'For this layer--> Covariance_matrix shape: {Covariance_matrix.shape}')


                # Step 4: Compute eigenvalues (TensorFlow 2.4+)
                eigenvalues = tf.linalg.eigvalsh(Covariance_matrix) # shape (c,)
                # make_logger.info(f'For this layer--> eigenvalues shape: {eigenvalues.shape}')


                # Step 5: Normalize eigenvalues to get probabilities
                prob_s = eigenvalues / tf.reduce_sum(eigenvalues)
                # make_logger.info(f'For this layer--> prob_s shape: {prob_s.shape}')


                # Step 6: Calculate expressivity_score using Shanon Entropy = -Seri[prob_s * log(prob_s)]
                # and Normalize the expressivity score throughout the channels

                score = -prob_s * tf.math.log(prob_s)
                # Check for NaN values
                if isinstance(score, tf.Tensor) and tf.math.reduce_any(tf.math.is_nan(score)):
                    expressivity_score = -np.inf
                    Normalized_expressivity_score = -np.inf
                else:
                    expressivity_score = tf.reduce_sum(score).numpy().item()
                    Normalized_expressivity_score = expressivity_score / tf.math.log(tf.cast(c, tf.float32)).numpy().item()

                # make_logger.info(f'For this layer--> Expressivity score: {expressivity_score}, Log (c): {tf.math.log(tf.cast(c, tf.float32))}, Normalized expressivity score: {Normalized_expressivity_score}')

                # Store expressivity score in dictionary
                exp_score_dict[layer_name] = {"layer_detail": layer_detail,
                                                "spatial_size": spatial_size, "num_channels": c,
                                                "log_c": tf.math.log(tf.cast(c, tf.float32)).numpy().item(),
                                                "expressivity_score": expressivity_score,
                                                "normalized_expressivity_score": Normalized_expressivity_score,
                                            }

                # make_logger.info(f'The score of {layer_name} was added to the dictionary')
                j += 1                    
            except Exception as e:
                # make_logger.warning(f'N.B: The score of {layer_name} was not added to the dictionary. Exception: {e}')
                continue


        make_logger.info(f"Expressivity scores was calculated for architecture {architectures[i]}")
        make_logger.info('-'*50)  

        # Save also some statistics of expressivity scores in the dataframe
        exp_score_df = get_statistics(exp_score_dict, show_exp_score=show_exp_score)

        all_exp_score_dict[architectures[i]] = exp_score_df

    return all_exp_score_dict