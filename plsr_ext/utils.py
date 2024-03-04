import numpy as np
from scipy.spatial.distance import cdist

def find_K_nearest_samples(A, B, K, metrics='euclidean'):
    """
    Find K samples in the 2-D matrix A with the minimum distance to the 
    1-D matrix B
    """
    distances = cdist(A, B.reshape(1, -1), metrics)  # Compute distances between rows of A and vector B
    closest_rows_indices = np.argsort(distances.flatten())[:K]  # Find indices of K closest rows
    return closest_rows_indices, A[closest_rows_indices], distances[closest_rows_indices].flatten()

def update_mean_std(new_X, new_Y, old_n_samples, old_X_mean, old_X_std, old_Y_mean, old_Y_std):
    """
    Update the mean and std values of training data according to new data

    Parameters
    ----------
    new_X : array-like of shape (n_samples, n_features)
        Original new training vectors, where `n_samples` is the number of new samples and
        `n_features` is the number of features.

    new_Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Orignal new target vectors, where `n_samples` is the number of new samples and
        `n_targets` is the number of response variables.
    """
    if new_X.ndim == 1:
        new_X = new_X.reshape(1, -1)

    if new_Y.ndim == 1:
        new_Y = new_Y.reshape(-1, 1)

    for i in range(new_X.shape[0]):
        n_samples = old_n_samples + 1
        new_x_mean = 1/n_samples * (new_X[i] + (n_samples - 1) * old_X_mean)
        new_y_mean = 1/n_samples * (new_Y[i] + (n_samples - 1) * old_Y_mean)
        new_x_std = np.sqrt((n_samples - 2) / (n_samples - 1) * old_X_std * old_X_std + \
                                1/n_samples * (new_X[i] - old_X_mean) *  (new_X[i] - old_X_mean))
        new_y_std = np.sqrt((n_samples - 2) / (n_samples - 1) * old_Y_std * old_Y_std + \
                                1/n_samples * (new_Y[i] - old_Y_mean) *  (new_Y[i] - old_Y_mean)) 
        
    return n_samples, new_x_mean, new_y_mean, new_x_std, new_y_std