# gappyfpca

`gappyfpca` is a simple, `numpy` and `scipy` based, Python package for generating a PCA representation of gappy functional data. It uses a pseudocovariance matrix of the gappy data to compute a first approximation for the principal components, a SLSQP algorithim is used to fit the coefficients. It can be run in parallel to speed up computation of the pseudocovariance and coefficients.

The method is iterated with reconstructed data until convergence criteria is met - less than 1\% average change in imputed data for at least X iterations or maximum number of iterations reached

## Installation

To install the package you can use `pip` after cloning. It is recommended to use a virtual environment to avoid conflicts.

	git clone
	cd gappyfpca
	pip install .

## Getting Started

- Functional data must be stored as discretised values, interpolated to the same spacing. NaN values represent missing data.
- Data stored in numpy array of dimention N x L where N in the number of data, and L is the length of data
- Data must be precleaned such that no data is fully empty and L = longest data ie no columns or rows with full NaNs, all data must span >0.5L

Compute PC components, coefficients and eigenvalues with do_gappyfpca(data,var_rat,max_iter=25,num_iter=10,iparallel=0)
	data - numpy array of dimension N x L
	var_rat - desired explained variance by returned components
	max_iter - maximum iterations
	num_iter - number of iterations convergence criteria must be satisfied for
	iparallel - flag to run in parallel with multiprocessing

 ## Citation

	 @article{hodgkin2025probabilistic,
	  title={Probabilistic Simulation of Aircraft Descent via a Hybrid Physics-Data Approach},
	  author={Hodgkin, Amy and Pepper, Nick and Thomas, Marc},
	  journal={TBC},
	  year={2025}
	}
