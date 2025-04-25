import time

import numpy as np

from gappyfpca.data_check import check_gappiness
from gappyfpca.eig import find_and_sort_eig, fpca_num_coefs
from gappyfpca.nancov import nancov
from gappyfpca.weights import fpca_weights


def reconstruct_func(
    fpca_mean: np.ndarray, fpca_comps: np.ndarray, fpca_coefs: np.ndarray, num_coefs: int | None = None
) -> np.ndarray:
    """
    Reconstruct the original data functions from FPCA components and coefficients.

    Parameters
    ----------
    fpca_mean : np.ndarray
        Mean function of the data. Shape is (L,).
    fpca_comps : np.ndarray
        Principal components Shape is (n_coefs + 1, L).
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    num_coefs : int, optional
        Number of coefficients to use for reconstruction. If None, all coefficients are used. Default is None.

    Returns
    -------
    np.ndarray
        Reconstructed data functions of shape (M, L).

    Notes
    -----
    The function reconstructs the original data functions by multiplying the FPCA coefficients
    with the principal components and adding the mean function.
    """
    if num_coefs is None:
        return np.matmul(fpca_coefs, fpca_comps) + fpca_mean

    if num_coefs > fpca_comps.shape[0]:
        num_coefs = fpca_comps.shape[0]

    return np.matmul(fpca_coefs[:, :num_coefs], fpca_comps[:num_coefs, :]) + fpca_mean


def fpca_initial(data: np.ndarray, iparallel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 1 (before iterative step) to compute FPCA components and coefficients for a set of gappy data functions.

    ** Note: Eigenvalues do not represent explained variance **

    Parameters
    ----------
    data : np.ndarray
        Array containing M discretized data functions, interpolated to the same length L, with NaN for missing data.
        Shape is (M, L).
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components, Shape is (n_coefs + 1, L), with the mean
        in row 0.
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    """
    # normalise data
    data_mean = np.nanmean(data, axis=0)
    data_norm = data - data_mean

    # calculate covariance matrix
    cov = nancov(data)

    # find and sort eigenvalues
    evalue, fpca_comps = find_and_sort_eig(cov)

    # retain number of coefficients for 100% explained variance
    n_coefs = fpca_num_coefs(evalue, 1, cov)
    fpca_comps = fpca_comps[:, :n_coefs]

    # compute PCA weights
    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    # stack mean and components for output
    fpca_comps = np.vstack((data_mean, fpca_comps.T))

    return fpca_comps, fpca_coefs


def fpca_update(
    data: np.ndarray, data_recon: np.ndarray, iparallel: int = 0
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
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components. Shape is (n_coefs + 1, L), with the mean
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

    # retain number of coefficients for 100% explained variance
    n_coefs = fpca_num_coefs(evalue, 1, cov)
    fpca_comps = fpca_comps[:, :n_coefs]

    # compute PCA weights
    fpca_coefs = fpca_weights(data_norm.T, fpca_comps, iparallel)

    # stack mean and components for output
    fpca_comps = np.vstack((data_mean_recon, fpca_comps.T))

    return fpca_comps, fpca_coefs, evalue


def l2_error(current: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """
    Calculate the change in reconstruction, L2

    Parameters
    ----------
    data_recon : np.ndarray
        Current reconstructed data.
    data_recon_prev : np.ndarray
        Previous reconstructed data.

    Returns
    -------
    relative_change : float
        Relative change in reconstruction.
    """
    return np.linalg.norm(prev - current) / np.linalg.norm(current)


def gappyfpca(
    data: np.ndarray,
    exp_var: float = 0.95,
    max_iter: int = 50,
    stable_iter: int = 4,
    tol: float = 5e-3,
    iparallel: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full iteration process to compute FPCA components and coefficients for a set of gappy data functions.
    Iterates for num_iter iterations with stopping criteria met or upper limit of max_iter.

    Parameters
    ----------
    data : np.ndarray
        Array containing M discretized data functions, interpolated to the same length L, with NaN for missing data.
        Shape is (M, L).
    exp_var : float, optional
        Explained variance to retain. This is the explained variance that the convergence will be tested at. Default is 0.95.
    max_iter : int, optional
        Maximum number of iterations. Default is 25.
    stable_iter : int, optional
        Number of iterations to achieve less than 1% change in reconstruction before stopping. Default is 5.
    tol : float, optional
        Tolerance for convergence. Default is 5e-3.
    iparallel : int, optional
        If 0, the calculation is done in series. If 1, the calculation is done in parallel. Default is 0.
    verbose : bool, optional
        If True, print progress messages. Default is True.

    Returns
    -------
    fpca_comps : np.ndarray
        Principal components, Shape is (n_coefs + 1, L), with the mean
        in row 0.
    fpca_coefs : np.ndarray
        Coefficients relating to data and PCs. Shape is (M, n_coefs).
    evalue : np.ndarray
        Eigenvalues of length n_coefs.
    data_dif : np.ndarray
        List of reconstruction differences at each iteration.

    Notes
    -----
    This function performs the full iteration process for FPCA on gappy data, using do_step1 and do_fpca_iterate
    functions.
    Iterates for num_iter iterations with stopping criteria met or upper limit of max_iter.
    """
    # do gappy fpca - calculate and iterate up to X iterations
    # stops iteration if 10 its of drag dif<=1% - I should maybe make this better
    start_time = time.time()

    if verbose:
        print("=" * 50)
        print("Gappy Functional PCA: Starting Analysis")
        print("=" * 50)
        print("Checking data validity...")

    check_gappiness(data)

    if verbose:
        print("-" * 30, "\n")
        print("Starting fPCA computation...")

    fpca_comps, fpca_coefs = fpca_initial(data, iparallel)
    # reconstruct data fully for iterative steps
    data_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)
    data_recon_test = np.copy(data_recon)
    end_time = time.time()

    if verbose:
        print(f"Time for initial step: {end_time - start_time:.2f} seconds\n")

    stable_count = 0
    it_count = 0
    data_dif = []

    if verbose:
        print("Entering iterative loop...\n")

    while stable_count < stable_iter and it_count < max_iter:
        if verbose:
            print(f"--- Iteration {it_count + 1}/{max_iter} ---")

        time_int = time.time()

        fpca_comps, fpca_coefs, evalue = fpca_update(data, data_recon, iparallel)

        data_recon_prev = np.copy(data_recon_test)
        data_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)
        num_recon = fpca_num_coefs(evalue, exp_var) if exp_var < 1 else None
        data_recon_test = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs, num_recon)

        # check if the reconstruction is stable on previous value with L2 norm
        recon_change = l2_error(data_recon, data_recon_prev)
        data_dif.append(recon_change)

        if recon_change < tol:
            stable_count += 1

            if verbose:
                print(
                    f"     Relative reconstruction is below tolerance: {recon_change:.2e} | Stable count: {stable_count}/{stable_iter}"
                )
        else:
            stable_count = 0
            if verbose:
                print(f"     Relative reconstruction is above tolerance: {recon_change:.2e}")

        it_count += 1

        end_time = time.time()
        if verbose:
            print(f"     Iteration time: {end_time - time_int:.3f} seconds")

    # crop to number of coefficients
    if num_recon is not None:
        fpca_comps = fpca_comps[: num_recon + 1, :]
        fpca_coefs = fpca_coefs[:, :num_recon]

    if verbose:
        print("=" * 50)
        print("Gappy fPCA Computation Finished")
        print(num_recon, "coefficients retained for", exp_var * 100, "% explained variance")
        print("=" * 50)
        print(f"Total iterations: {it_count}/{max_iter}")
        print(f"Stable iterations: {stable_count}/{stable_iter}")
        print(f"Final relative reconstruction change: {data_dif[-1]:.2e}")
        print(f"Total computation time: {end_time - start_time:.2f} seconds")

    return fpca_comps, fpca_coefs, evalue, data_dif
