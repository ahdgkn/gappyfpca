import time

import numpy as np

from gappyfpca.eig import find_and_sort_eig, fpca_num_coefs
from gappyfpca.nancov import nancov
from gappyfpca.weights import fpca_weights


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
