from strlearn.utils import scores_to_cummean
import numpy as np

def calculate_cummean(scores):


    n_estimators = len(np.unique(scores[:, 0]))
    n_chunks = len(np.unique(scores[:, 1]))
    n_metrics = scores.shape[1] - 2

    reshaped_scores = scores[:, 2:].reshape(n_estimators, n_chunks, n_metrics)    
    cummean_scores = scores_to_cummean(reshaped_scores)

    flattened_scores = cummean_scores.reshape(-1, n_metrics)          
    restored_data = np.column_stack((scores[:, 0], scores[:, 1], flattened_scores))

    return restored_data
