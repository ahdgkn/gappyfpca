from multiprocessing import Pool

import numpy as np

def nancov(A: np.ndarray,iparallel=0) -> np.ndarray:

    """
    Calculate a pseudocovariance matrix for a dataset, ignoring NaN values.
    Parameters
    ----------
    A : np.ndarray
        Interpolated data array of shape (N, L), where each row represents one data function.

    Returns
    -------
    np.ndarray
        Approximated covariance matrix of shape (L, L). cov = 1/N A.T . A

    Notes
    -----
    The function computes the covariance matrix by normalizing the input data and then
    calculating the dot product for each pair of features, ignoring NaN values.
    """

    # Create a binary mask for valid (non-NaN) values
    nan_mask = ~np.isnan(A)

    #normalise data
    A_norm = A - np.nanmean(A, axis=0)
    # Replace NaNs with zeros for dot product computation
    A_filled = np.where(nan_mask, A_norm, 0)

    # Compute the dot product of the data matrix
    cov_num = np.dot(A_filled.T, A_filled)

    # Compute the number of valid (non-NaN) pairs for each covariance entry
    N = np.dot(nan_mask.astype(int).T, nan_mask.astype(int)) 
    # Avoid division by zero
    if np.any(N == 0):
        raise ValueError("Some covariance entries have no valid data points. Check your data with data_gappiness function.")

    # Compute the weighted covariance matrix
    cov = cov_num / N

    return cov