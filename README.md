<p align="center">
  <img src="gappyfpca_logo.png" alt="gappyfpca logo" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/amyhodgkin/gappyfpca/actions"><img src="https://github.com/amyhodgkin/gappyfpca/actions/workflows/ci-gappyfpca.yml/badge.svg"></a>
   <a href="https://arxiv.org/abs/2504.02529"><img src="https://img.shields.io/badge/arXiv-2504.02529-b31b1b.svg"></a>
  <img src="https://img.shields.io/badge/dependencies-numpy%2C%20scipy%2C%20matplotlib-blue">
</p>

`gappyfpca` is a simple, `numpy` and `scipy` based, Python package for generating a PCA representation of gappy functional data. It uses a pseudocovariance matrix of the gappy data to compute a first approximation for the principal components and a SLSQP minimisation algorithim to project the gappy data onto these. Subsequent iterations find the covariance from the reconstructed data and repeat the minimisation step to update the coefficients with the gappy data. The minimisation step can be run in parallel to speed up computation of the coefficients.

This package was developed to create a low-order representation of aircraft trajectories for generative modelling. If you use this package please cite the paper referenced below.

## Workflow

The package implements a fPCA algorithm with a pseudocovariance calculated with gappy data to replace a covariance matrix for the first step. PCA weights are fitted with a optimisation function. Subsequent steps use reconstructed data to calculate the full covariance.

1. Check suitability of data for method, clean data to remove 'too much' gappiness if needed
2. 'fpca_initial' - Inital fPCA computation using gappy data
	1. 'nancov' - compute psuedocovariance
	2. 'eig_decomp' - returns sorted eigenvalue decomposition
	3. 'fpca_num_coefs' - ensure only valid components are retained
	4. 'fpca_weights' - compute fPCA weights/coefficients with sequential minimisation
	5. return components and coefficients
3. 'reconstruct_function' - Use fPCA representation to reconstruct (impute) missing data
4. Enter iterative process, repeat until convergence or maximum iterations reached
	1. 'fpca_update' - update fPCA computation with reconstructed data
		1. Like 'fpca_initial' but step 1 is replaced with np.cov
	2. 'reconstruct_function'
	3. 'check_convergence' - Compare L2 error of current function reconstruction with previous, check for stability
	4. If reconstruction error is less than specified tolerance for X (stable_iter) number of iterations OR total number of iterations is maximum iterations, exit loop. Else repeeat 4.1-4.4
5. Return fPCA components, coefficients, eigenvalues and convergence stats

INCLUDE WORKFLOW DIAGRAM HERE

## Installation

To install the package locally you can use `pip` after cloning. It is recommended to use a virtual environment to avoid conflicts.

	git clone https://github.com/amyhodgkin/gappyfpca.git
	cd gappyfpca
	pip install .

The package can also be installed in editable mode with:

	pip install -e .

If you wish to run tests, you can install the dependencies required for these with:

	pip install .[test]
 
## Getting Started

A example notebook 'get_started.ipynb' is provided with details the use of the package through some simple synthetically generated data. Considerations when using the package are discussed within the notebook.

 ## Citation

	@article{hodgkin2025probabilistic,
	  title={Probabilistic Simulation of Aircraft Descent via a Hybrid Physics-Data Approach},
	  author={Hodgkin, Amy and Pepper, Nick and Thomas, Marc},
	  journal={arXiv preprint arXiv:2504.02529},
	  year={2025}
	}
