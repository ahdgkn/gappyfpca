import numpy as np
from scipy.linalg import eigh


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
    return np.cumsum(eigenvalues) / np.sum(eigenvalues)


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

    return np.argmax(exp_var - var_rat >= 0) if var_rat != 1 else min(len(data[:, 0]), len(data[0, :]) - 1)
