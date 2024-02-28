import numpy as np
from scipy.spatial.distance import cdist

def find_K_nearest_samples(A, B, K, metrics):
    """
    Find K samples in the 2-D matrix A with the minimum distance to the 
    1-D matrix B
    """
    distances = cdist(A, B.reshape(1, -1), metrics)  # Compute distances between rows of A and vector B
    closest_rows_indices = np.argsort(distances.flatten())[:K]  # Find indices of K closest rows
    return closest_rows_indices, A[closest_rows_indices]