from multiprocessing import Pool

import numpy as np


def nancov_series(A: np.ndarray) -> np.ndarray:
    """
    Calculate an approximated covariance matrix for a dataset, ignoring NaN values.

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
    n, p = A.shape
    Anorm = A - np.nanmean(A, axis=0)
    cov = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            dot = Anorm[:, i] * Anorm[:, j]
            norm = len(dot[~np.isnan(dot)])

            cov[i, j] = np.nansum(dot) / (norm)

    return cov


def nancov_helper(args: tuple[np.ndarray, np.ndarray]) -> float:
    """
    Helper function for parallel covariance calculation.

    Parameters
    ----------
    args : tuple[np.ndarray, np.ndarray]
        A tuple containing two 1D numpy arrays representing the data vectors for which
        the covariance element is to be calculated.

    Returns
    -------
    float
        The calculated covariance element for the given pair of data vectors.

    Notes
    -----
    This function calculates one element of the dot product for the covariance matrix,
    ignoring NaN values in the input data vectors.
    """
    A1, A2 = args
    dot = A1 * A2
    norm = len(dot[~np.isnan(dot)])
    return np.nansum(dot) / norm


def nancov_parallel(A: np.ndarray) -> np.ndarray:
    """
    Calculate an approximated covariance matrix for a dataset, ignoring NaN values, , using parallel processing.

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
    n, p = A.shape
    Anorm = A - np.nanmean(A, axis=0)
    cov = np.zeros((p, p))
    args_list = [(Anorm[:, i], Anorm[:, j]) for i in range(p) for j in range(p)]
    with Pool() as pool:
        results = pool.map(nancov_helper, args_list)
    for i in range(p):
        for j in range(p):
            cov[i, j] = results[i * p + j]

    return cov


def nancov(A: np.ndarray, iparallel: int = 0) -> np.ndarray:
    """
    Calculate an approximated covariance matrix for a dataset, ignoring NaN values.

    Parameters
    ----------
    A : np.ndarray
        Interpolated data array of shape (N, L), where each row represents one data function.
    iparallel : int
        If 0, the calculation is done in series. If 1, the calculation is done in parallel.

    Returns
    -------
    np.ndarray
        Approximated covariance matrix of shape (L, L). cov = 1/N A.T . A

    Notes
    -----
    The function computes the covariance matrix by normalizing the input data and then
    calculating the dot product for each pair of features, ignoring NaN values.
    """
    # if iparallel=0 do in series, otherwise parallel
    if iparallel == 0:
        return nancov_series(A)
    return nancov_parallel(A)
