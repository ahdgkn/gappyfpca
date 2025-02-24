import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


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


def find_and_sort_eig(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform eigenvalue decomposition and sort the results in descending order of eigenvalue size.

    Parameters
    ----------
    A : np.ndarray
        Input array for which eigenvalue decomposition is to be performed.

    Returns
    -------
    evalue : np.ndarray
        Eigenvalues sorted in descending order.
    evector : np.ndarray
        Eigenvectors sorted in descending order of corresponding eigenvalues.
        These are equal to the principal components if A is a covariance matrix.

    Notes
    -----
    The function uses the `eigh` method for eigenvalue decomposition, which is suitable for
    symmetric or Hermitian (conjugate symmetric) matrices.
    """

    evalue, evector = eigh(A)

    ##3. sort eigenvalues in descending order
    sorted_indices = np.argsort(evalue)[::-1]
    evalue = evalue[sorted_indices]
    evector = evector[:, sorted_indices]

    return evalue, evector


def fpca_exp_var(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Returns cumulative explained variance of principal components in descending order.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Sorted eigenvalues corresponding to principal components.

    Returns
    -------
    np.ndarray
        Array with the cumulative explained variance of principal components.

    Notes
    -----
    The function calculates the cumulative explained variance by summing the eigenvalues
    and dividing by the total sum of eigenvalues.
    """
    exp_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return exp_var


def fpca_num_coefs(evalue: np.ndarray, var_rat: float, data: np.ndarray = None) -> int:
    """
    Calculate the number of coefficients to retain for an explained variance of var_rat.

    Parameters
    ----------
    evalue : np.ndarray
        Array of eigenvalues.
    var_rat : float
        Fraction of explained variance to retain, between 0 and 1.
    data : np.ndarray, optional
        Data array, required if var_rat is 1 to determine the limits of max coefficients based on amount of data.

    Returns
    -------
    int
        Number of coefficients to retain.
    """

    if var_rat == 1 and data is None:
        raise ValueError("Must provide data array for var_rat=1")
    exp_var = np.array([0])
    exp_var = np.hstack((exp_var, fpca_exp_var(evalue)))

    n_coefs = np.argmax(exp_var - var_rat >= 0) if var_rat != 1 else min(len(data[:, 0]), len(data[0, :]) - 1)

    return n_coefs


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
    init_weight = 0
    fpca_weights = np.zeros(n_coefs)
    for i in range(n_coefs):
        fpca_comp = PCs[:, i]
        mask = np.isnan(data_func)
        fpca_comp_masked = fpca_comp[~mask]
        data_func_masked = data_func[~mask]
        result = minimize(sum_sq_error, init_weight, args=(data_func_masked, fpca_comp_masked), method="SLSQP")
        fpca_weights[i] = result.x
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


def do_step1(data: np.ndarray, var_rat: float, iparallel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 1 (before iterative step) to compute FPCA components and coefficients for a set of gappy data functions.

    ** Note: Eigenvalues do not represent explained variance **

    Parameters
    ----------
    data : np.ndarray
        Array containing M discretized data functions, interpolated to the same length L, with NaN for missing data.
        Shape is (M, L).
    var_rat : float
        Desired explained variance to retain components, between 0 and 1.
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components, with the number of components given by var_rat. Shape is (n_coefs + 1, L), with the mean
        in row 0.
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    """
    # normalise data
    data_mean = np.nanmean(data, axis=0)
    data_norm = data - data_mean

    # calculate covariance matrix
    cov = nancov(data, iparallel)

    # find and sort eigenvalues
    evalue, fpca_comps = find_and_sort_eig(cov)

    # retain number of coefficients for desired explained variance
    n_coefs = fpca_num_coefs(evalue, var_rat, data_norm)
    fpca_comps = fpca_comps[:, :n_coefs]

    # compute PCA weights
    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    # stack mean and components for output
    fpca_comps = np.vstack((data_mean, fpca_comps.T))

    return fpca_comps, fpca_coefs


def reconstruct_func(fpca_mean: np.ndarray, fpca_comps: np.ndarray, fpca_coefs: np.ndarray) -> np.ndarray:
    """
    Reconstruct the original data functions from FPCA components and coefficients.

    Parameters
    ----------
    fpca_comps : np.ndarray
        Principal components, including the mean in the first row. Shape is (n_coefs + 1, L).
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).

    Returns
    -------
    np.ndarray
        Reconstructed data functions of shape (M, L).

    Notes
    -----
    The function reconstructs the original data functions by multiplying the FPCA coefficients
    with the principal components and adding the mean function.
    """
    return np.matmul(fpca_coefs, fpca_comps) + fpca_mean


def do_fpca_iterate(
    data: np.ndarray, data_recon: np.ndarray, var_rat: float, iparallel: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative step to compute FPCA components and coefficients for a set of gappy data functions from reconstructed
    data.

    Parameters
    ----------
    data : np.ndarray
        Array containing M discretized data functions, interpolated to the same length L, with NaN for missing data.
        Shape is (M, L).
    data_recon : np.ndarray
        Array containing reconstructed data functions, with no missing data. Shape is (M, L).
    var_rat : float
        Desired explained variance to retain components, between 0 and 1.
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components, with the number of components given by var_rat. Shape is (n_coefs + 1, L), with the mean
        in row 0.
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    evalue : np.ndarray
        Eigenvalues of length n_coefs.

    """
    # normalise data with reconstructed data mean
    data_mean_recon = np.nanmean(data_recon, axis=0)
    data_norm = data - data_mean_recon

    # calculate covariance matrix of reconstructed data
    cov = np.cov(data_recon, bias=True, rowvar=False)

    # find and sort eigenvalues
    evalue, fpca_comps = find_and_sort_eig(cov)

    # retain number of coefficients for desired explained variance
    n_coefs = fpca_num_coefs(evalue, var_rat, data_norm)
    fpca_comps = fpca_comps[:, :n_coefs]

    # compute PCA weights
    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    # stack mean and components for output
    fpca_comps = np.vstack((data_mean_recon, fpca_comps.T))

    return fpca_comps, fpca_coefs, evalue


def gappyfpca(
    data: np.ndarray, var_rat: float, max_iter: int = 25, num_iter: int = 10, iparallel: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full iteration process to compute FPCA components and coefficients for a set of gappy data functions.
    Iterates for num_iter iterations with stopping criteria met or upper limit of max_iter.

    Parameters
    ----------
    data : np.ndarray
        Array containing M discretized data functions, interpolated to the same length L, with NaN for missing data.
        Shape is (M, L).
    var_rat : float
        Desired explained variance to retain components, between 0 and 1.
    max_iter : int, optional
        Maximum number of iterations. Default is 25.
    num_iter : int, optional
        Number of iterations to achieve less than 1% change in reconstruction before stopping. Default is 10.
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components, with the number of components given by var_rat. Shape is (n_coefs + 1, L), with the mean
        in row 0.
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    evalue : np.ndarray
        Eigenvalues of length n_coefs.
    run_stat : np.ndarray
        Array of convergence stats, where row 1 is the difference between data_recon_i and data_recon_i-1, and row 2 is
        coef[0,0].

    Notes
    -----
    This function performs the full iteration process for FPCA on gappy data, using do_step1 and do_fpca_iterate
    functions.
    Iterates for num_iter iterations with stopping criteria met or upper limit of max_iter.
    """
    # do gappy fpca - calculate and iterate up to X iterations
    # stops iteration if 10 its of drag dif<=1% - I should maybe make this better
    start_time = time.time()
    fpca_comps, fpca_coefs = do_step1(data, var_rat, iparallel)
    data_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)
    end_time = time.time()
    print("Step 1, time:", end_time - start_time)

    it_count = 0
    it_total = 0
    data_dif = []
    coef1 = []
    while it_count < num_iter and it_total < max_iter:
        time1 = time.time()
        print("Iteration ", it_total + 1)

        fpca_comps, fpca_coefs, evalue = do_fpca_iterate(data, data_recon, var_rat, iparallel)

        data_recon_old = np.copy(data_recon)
        data_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)

        x = np.mean(np.abs((data_recon - data_recon_old) / data_recon_old))
        data_dif.append(x)
        coef1.append(np.abs(fpca_coefs[0, 0]))
        if x <= 0.01:
            it_count += 1
        else:
            it_count = 0

        it_total += 1

        end_time = time.time()
        print("Time: ", end_time - time1)

    run_stat = np.vstack((data_dif, coef1))

    return fpca_comps, fpca_coefs, evalue, run_stat
