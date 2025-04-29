from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize


def sum_sq_error(weight: float, data_func: np.ndarray, fpca_comp: np.ndarray) -> float:
    """
    Calculate the sum of squared error for optimizing the weight.

    Parameters
    ----------
    weight : float
        Weight to optimize (w_ik).
    data_func : np.ndarray
        Functional data (i) to fit the weight for.
    fpca_comp : np.ndarray
        Principal component (k) to fit the weight for.

    Returns
    -------
    float
        Sum of squared error between the data function and weight * principal component.

    Notes
    -----
    The data_func and fpca_comp must be of the same length and contain no missing data.
    """
    fitted_component = weight * fpca_comp
    return np.sum((data_func - fitted_component) ** 2)


def process_weights(args: tuple[int, np.ndarray, np.ndarray, int]) -> np.ndarray:
    """
    Function to process weight optimization for gappy functions.

    Parameters
    ----------
    args : tuple[int, np.ndarray, np.ndarray, int]
        A tuple containing the following elements:
        - j (int): Index of the data function to fit.
        - data_func (np.ndarray): jth data function, of length L.
        - PCs (np.ndarray): Principal components of shape (M, L) where M is the number of principal components.
        - n_coefs (int): Number of coefficients to compute, n_coefs <= min(M,L-1).

    Returns
    -------
    np.ndarray
        Array of optimized weights for the given data function and principal components.

    Notes
    -----
    This function optimizes the weights for the given data function and principal components
    by minimizing the sum of squared errors.
    """
    j, data_func, PCs, n_coefs = args
    init_weight = 0.0
    fpca_weights = np.zeros(n_coefs)
    mask = np.isnan(data_func)
    for i in range(n_coefs):
        fpca_comp = PCs[:, i]
        fpca_comp_masked = fpca_comp[~mask]
        data_func_masked = data_func[~mask]
        result = minimize(sum_sq_error, init_weight, args=(data_func_masked, fpca_comp_masked), method="SLSQP")
        fpca_weights[i] = result.x[0] if isinstance(result.x, np.ndarray) else result.x
        data_func = data_func - fpca_weights[i] * fpca_comp
    return fpca_weights


def fpca_weights_parallel(data_funcs: np.ndarray, PCs: np.ndarray) -> np.ndarray:
    """
    Compute the full set of weights (w_ij) for gappy data functions and principal components using SLSQP minimization in
    parallel.

    Parameters
    ----------
    data_funcs : np.ndarray
        Array of M data functions with length L (shape M x L), containing NaN for gappy data.
    PCs : np.ndarray
        Array of N principal components with length L (shape N x L), where N is the number of coefficients to compute.

    Returns
    -------
    np.ndarray
        Array containing weights of shape M x N.
    """
    n, p = data_funcs.shape
    n, n_coefs = PCs.shape
    fpca_weights = np.zeros((p, n_coefs))
    with Pool() as pool:
        args_list = [(j, data_funcs[:, j], PCs, n_coefs) for j in range(p)]
        results = pool.map(process_weights, args_list)

    for args, weight in zip(args_list, results, strict=False):
        j, _, _, _ = args
        fpca_weights[j, :] = weight

    return fpca_weights


def fpca_weights_series(data_funcs: np.ndarray, PCs: np.ndarray) -> np.ndarray:
    """
    Compute the full set of weights (w_ij) for gappy data functions and principal components using SLSQP minimization.

    Parameters
    ----------
    data_funcs : np.ndarray
        Array of M data functions with length L (shape M x L), containing NaN for gappy data.
    PCs : np.ndarray
        Array of N principal components with length L (shape N x L), where N is the number of coefficients to compute.

    Returns
    -------
    np.ndarray
        Array containing weights of shape M x N.
    """

    n, p = data_funcs.shape
    n, n_coefs = PCs.shape
    fpca_weights = np.zeros((p, n_coefs))
    for j in range(p):
        args = (j, data_funcs[:, j], PCs, n_coefs)
        fpca_weights[j, :] = process_weights(args)

    return fpca_weights


def fpca_weights(data_funcs: np.ndarray, PCs: np.ndarray, iparallel: int = 0) -> np.ndarray:
    """
    Compute the full set of weights (w_ij) for gappy data functions and principal components using SLSQP minimization.

    Parameters
    ----------
    data_funcs : np.ndarray
        Array of M data functions with length L (shape M x L), containing NaN for gappy data.
    PCs : np.ndarray
        Array of N principal components with length L (shape N x L), where N is the number of coefficients to compute.
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    np.ndarray
        Array containing weights of shape M x N.
    """
    if iparallel == 0:
        return fpca_weights_series(data_funcs, PCs)
    return fpca_weights_parallel(data_funcs, PCs)
