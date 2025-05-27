import numpy as np
import pytest

from gappyfpca.data_check import check_gappiness
from gappyfpca.fpca import gappyfpca, reconstruct_func

@pytest.mark.parametrize("iparallel", [0, 1])
def test_gappyfpca_integration(iparallel):
    """Integration test for gappyfpca accuracy (serial and parallel)."""

    # generate synthetic dataset to test
    # Parameters
    M = 1000  # Number of functions
    L = 50   # Length of each function

    # Sinusoidal patterns
    np.random.seed(42) # Ensure reproducibility
    x = np.linspace(0, 2 * np.pi, L)
    functions = np.array([np.polyval(np.random.uniform(-1, 1, size=1), np.linspace(-1, 1, L)) 
                      for _ in range(M)])

    data = np.copy(functions)
    # Artificially make it gappy
    for i in range(M):
        num_nans = np.random.randint(0, L // 2)
        nan_indices = np.random.choice(L, num_nans, replace=False)
        data[i, nan_indices] = np.nan

    # Check data validity before running gappyfpca
    check_gappiness(data)

    # Run gappyfpca
    fpca_comps, fpca_coefs, evalue, run_stat = gappyfpca(data, exp_var=0.95, max_iter=10, stable_iter=5, tol = 5e-3, iparallel=iparallel)

    # Impute missing data
    function_recon = reconstruct_func(fpca_comps[0, :], fpca_comps[1:, :], fpca_coefs)

    if np.any(np.isnan(function_recon)):
         pytest.fail(f"Reconstructed function contains NaNs for iparallel={iparallel}")

    # Calculate mean absolute error across all points

    # Assert that the mean absolute error is below a threshold
    assert run_stat[-1] < 1e-4, f"Mean reconstruction error {run_stat[-1]} is too high for iparallel={iparallel}"