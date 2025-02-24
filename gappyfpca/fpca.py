import time
from multiprocessing import Pool
from typing import tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize


def nancov_series(A: np.ndarray) -> np.ndarray:
    """Calculate a covariance approximation skipping NaN values in series

    Parameters
    -----------
    A: array
        Interpolated data, N x L, each row is one data function

    Returns
    --------
    cov = 1/N A.T . A
        Approximated covariance
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
    """Helper for parallel covariance function
    calculate one element of dot product
    """
    A1, A2 = args
    dot = A1 * A2
    norm = len(dot[~np.isnan(dot)])
    return np.nansum(dot) / norm


def nancov_parallel(A: np.ndarray) -> np.ndarray:
    """Calculate a covariance approximation skipping NaN values in parallel

    Parameters
    -----------
    A: array
        Interpolated data, N x L, each row is one data function

    Returns
    --------
    cov = 1/N A.T . A
        Approximated covariance
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
    """Calculate a covariance approximation skipping NaN values

    Parameters
    -----------
    A: array
        Interpolated data, N x L, each row is one data function
    iparallel: int
        0: series, 1: parallel

    Returns
    --------
    cov = 1/N A.T . A
        Approximated covariance
    """
    # if iparallel=0 do in series, otherwise parallel
    if iparallel == 0:
        return nancov_series(A)
    return nancov_parallel(A)


def find_and_sort_eig(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalue decomposition and sort in descending evalue size

    Parameters
    -----------
    A: array

    Returns
    --------
    evalue:
        Eigenvalues sorted in descending size order
    evector:
        Eigenvectors sorted in descending evalue size order
        equal to principal components if A is covariance matrix
    """

    evalue, evector = eigh(A)

    ##3. sort eigenvalues in descending order
    sorted_indices = np.argsort(evalue)[::-1]
    evalue = evalue[sorted_indices]
    evector = evector[:, sorted_indices]

    return evalue, evector


def fpca_exp_var(eigenvalues: np.ndarray) -> np.ndarray:
    """Returns cumulative explained variance of PC in descending order
    Parameters
    -----------
    eigenvalues:
        sorted eigenvalues corresponding to principal components

    Returns
    --------
    exp_var: array
        array with the cumulative explained variance of PC with eigenvalues as input
    """
    exp_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return exp_var


def fpca_num_coefs(evalue: np.ndarray, var_rat: float, data: np.ndarray = None) -> int:
    """Calculate the number of coefficients to retain for an explained variance of var_rat
    Parameters
    -----------
    evalue:
        array of evalues
    var_rat: 0 to 1
        fraction of explained variance to retain
    data:
        data array, if var_rat=1 to give limits of max coefs
    Returns
    --------
    n_coefs:
        number of coefficients to retain
    """

    if var_rat == 1 and data is None:
        raise ValueError("Must provide data array for var_rat=1")
    exp_var = np.array([0])
    exp_var = np.hstack((exp_var, fpca_exp_var(evalue)))

    n_coefs = np.argmax(exp_var - var_rat >= 0) if var_rat != 1 else min(len(data[:, 0]), len(data[0, :]) - 1)

    return n_coefs


def sum_sq_error(weight: float, data_func: np.ndarray, fpca_comp: np.ndarray) -> float:
    """Sum of squared error for optimising weight ik
    Parameters
    -----------
    weight:
        weight to optimise w_ik
    data_func:
        functional data, i, to fit weight for
    fpca_comp:
        principal component, k, to fit weight for

    data_func and fpca_comp must be same length and with no missing data
    Returns
    --------
    Sum squared error between data function and weight * PC
    """
    fitted_component = weight * fpca_comp
    return np.sum((data_func - fitted_component) ** 2)


def process_weights(args: tuple[int, np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Function to process weight optimisation for gappy functions
    Parameters
    -----------
    j:
        index of data function to fit
    data_func:
        jth data function, of length L
    PCs:
        principal component of shape M x L where M is number of PC
    n_coefs:
        number of coeficients to compute, n_coefs<=M
    Returns:
    --------
    fpca_weights:
        array of length n_coef containing the weights for data func j
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

    """Compute full set of w_ij for gappy data functions and PCs using SQLSP minimisation, parallel
    Parameters
    -----------
    data_funcs:
        array of M  data_funcs with length L - M x L, NaN for gappy data
    PCs:
        array of N principal components with length L - N x L, N=n_coefs to compute
    Returns
    --------
    fpca_weights:
        array containing weights of shape M x N
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
    """Compute full set of w_ij for gappy data functions and PCs using SQLSP minimisation, series
    Parameters
    -----------
    data_funcs:
        array of M  data_funcs with length L - M x L, NaN for gappy data
    PCs:
        array of N principal components with length L - N x L, N=n_coefs to compute
    Returns
    --------
    fpca_weights:
        array containing weights of shape M x N
    """

    n, p = data_funcs.shape
    n, n_coefs = PCs.shape
    fpca_weights = np.zeros((p, n_coefs))
    for j in range(p):
        args = (j, data_funcs[:, j], PCs, n_coefs)
        fpca_weights[j, :] = process_weights(args)

    return fpca_weights


def fpca_weights(data_funcs: np.ndarray, PCs: np.ndarray, iparallel: int = 0) -> np.ndarray:
    """Compute full set of w_ij for gappy data functions and PCs using SQLSP minimisation
    Parameters
    -----------
    data_funcs:
        array of M  data_funcs with length L - M x L, NaN for gappy data
    PCs:
        array of N principal components with length L - N x L, N=n_coefs to compute
    iparallel: int
        0: series, 1: parallel
    Returns
    --------
    fpca_weights:
        array containing weights of shape M x N
    """
    if iparallel == 0:
        return fpca_weights_series(data_funcs, PCs)
    return fpca_weights_parallel(data_funcs, PCs)


def do_gappyfpca(data: np.ndarray, var_rat: float, iparallel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute fpca components and coefficients for a set of gappy data functions
    Step 1 of iterative process
    ** evalues do not represent explained variance **
    Parameters
    -----------
    data:
        array containing M discretised data functions, integrated to same length L, NaN for missing data. shape M x L
    var_rat:
        desired explained variance to retain components: 0 to 1
    iparallel:
        0: series, 1: parallel

    Returns
    --------
    fpca_comps:
        principal components, n_coefs given by var_rat. shape n_coefs+1 x L. mean giving in row 0
    fpca_coefs
        coefficients relating to data and PCs, shape M x n_coefs
    """

    data_mean = np.nanmean(data, axis=0)

    data_norm = data - data_mean

    cov = nancov(data, iparallel)

    evalue, fpca_comps = find_and_sort_eig(cov)

    n_coefs = fpca_num_coefs(evalue, var_rat, data_norm)

    fpca_comps = fpca_comps[:, :n_coefs]

    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    fpca_comps = np.vstack((data_mean, fpca_comps.T))

    return fpca_comps, fpca_coefs


def reconstruct_func(fpca_mean: np.ndarray, fpca_comps: np.ndarray, fpca_coefs: np.ndarray) -> np.ndarray:
    """Construct data functions from fPCA mean, components and coefficients
    Parameters
    -----------
    fpca_mean:
        mean of data, length L
    fpca_comps:
        PCs, size N x L
    fpca_coefs:
        coefficients, size M x N
    N is number of PCs desired for reconstruction
    Returns
    -------
    func_recon:
        reconstructed data, size M x L
    """
    func_recon = np.matmul(fpca_coefs, fpca_comps) + fpca_mean

    return func_recon


def do_fpca_iterate(data: np.ndarray, data_recon: np.ndarray, var_rat: float, iparallel: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative step to compute fpca components and coefficients for a set of gappy data functions from reconstructed data
    Step 2 -> of iterative process
    Parameters
    -----------
    data:
        array containing M discretised data functions, integrated to same length L, NaN for missing data. shape M x L
    data_recon
        array containing reconstructed data functions, no missing data, shape M x L
    var_rat:
        desired explained variance to retain components: 0 to 1
    iparallel:
        0: series, 1: parallel

    Returns
    --------
    fpca_comps:
        principal components, n_coefs given by var_rat. shape n_coefs+1 x L. mean giving in row 0
    fpca_coefs
        coefficients relating to data and PCs, shape M x n_coefs
    evalue:
        eigenvalues length n_coefs
    """

    data_mean_recon = np.nanmean(data_recon, axis=0)

    data_recon_norm = data_recon - data_mean_recon

    data_mean = np.nanmean(data, axis=0)

    data_norm = data - data_mean_recon

    cov = np.cov(data_recon, bias=True, rowvar=False)

    evalue, fpca_comps = find_and_sort_eig(cov)

    n_coefs = fpca_num_coefs(evalue, var_rat, data_norm)

    fpca_comps = fpca_comps[:, :n_coefs]

    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    fpca_comps = np.vstack((data_mean_recon, fpca_comps.T))

    return fpca_comps, fpca_coefs, evalue


def gappyfpca(data: np.ndarray, var_rat: float, max_iter: int = 25, num_iter: int = 10, iparallel: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full iteration process to compute fpca components and coefficients for a set of gappy data functions
    Iterates for num_iter iterations with stopping criteria met or upper limit of max_iter
    Parameters
    -----------
    data:
        array containing M discretised data functions, integrated to same length L, NaN for missing data. shape M x L
    var_rat:
        desired explained variance to retain components: 0 to 1
    max_iter:
        maximum number of iterations
    num_iter:
        number of iterations to achieve less than 1% change in reconstruction before stopping
    iparallel:
        0: series, 1: parallel

    Returns
    --------
    fpca_comps:
        principal components, n_coefs given by var_rat. shape n_coefs+1 x L. mean giving in row 0
    fpca_coefs
        coefficients relating to data and PCs, shape M x n_coefs
    evalue:
        eigenvalues length n_coefs
    run_stat
        array of convergence stats, row 1 is difference between data_recon_i and data_recon_i-1 and row 2 is coef[0,0]
    """
    # do gappy fpca - calculate and iterate up to X iterations
    # stops iteration if 10 its of drag dif<=1% - I should maybe make this better
    start_time = time.time()
    fpca_comps, fpca_coefs = do_gappyfpca(data, var_rat, iparallel)
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
