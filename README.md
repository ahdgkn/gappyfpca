# gappyfpca

`gappyfpca` is a simple, `numpy` and `scipy` based, Python package for generating a PCA representation of gappy functional data. It uses a pseudocovariance matrix of the gappy data to compute a first approximation for the principal components and a SLSQP minimisation algorithim to fit the coefficients. Subsequent iterations find the covariance from the reconstructed data and repeat the minimisation step to fit the coefficients with the gappy data. It can be run in parallel to speed up computation of the pseudocovariance and coefficients.

The method is iterated until there is either 1) a less than 1\% average change in imputed data for at least X iterations or 2) the maximum number of iterations is reached.

This package was developed to create a low-order representation of aircraft trajectories for generative modelling. If you use this package please cite the paper referenced below.

## Installation

To install the package you can use `pip` after cloning. It is recommended to use a virtual environment to avoid conflicts.

	git clone
	cd gappyfpca
	pip install .

Dependencies required for included tests can be downloaded using

	pip install xxx
 
## Getting Started

There is an example notebook provided in `tests` which generates a set of synthetic data, makes it gappy, and runs the function `gappyfpca`

Some considerations when using the package:

- Functional data must be stored as discretised values, interpolated to the same spacing. NaN values represent missing data.
- Data stored in numpy array of dimention N x L where N in the number of data, and L is the length of data
- Data must be precleaned such that no data is fully empty and L = longest data ie no columns or rows with full NaNs, all data must span >0.5L **hard limit
- Compute PC components, coefficients and eigenvalues with gappyfpca(data,var_rat,max_iter=25,num_iter=10,iparallel=0)
	- data - numpy array of dimension N x L
	- var_rat - desired explained variance by returned components
	- max_iter - maximum iterations
	- num_iter - number of iterations convergence criteria must be satisfied for
	- iparallel - flag to run in parallel with multiprocessing

The package is suitable for small datasets. As an example, the algorithm takes 75 s per iteration to compute the principal components for a set of 5000 functions of length 100 on 16 cores.

 ## Citation

	 @article{hodgkin2025probabilistic,
	  title={Probabilistic Simulation of Aircraft Descent via a Hybrid Physics-Data Approach},
	  author={Hodgkin, Amy and Pepper, Nick and Thomas, Marc},
	  journal={TBC},
	  year={2025}
	}
