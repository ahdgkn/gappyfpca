# gappyfpca

`gappyfpca` is a simple, `numpy` and `scipy` based, Python package for generating a PCA representation of gappy functional data. It uses a pseudocovariance matrix of the gappy data to compute a first approximation for the principal components and a SLSQP minimisation algorithim to fit the coefficients. Subsequent iterations find the covariance from the reconstructed data and repeat the minimisation step to fit the coefficients with the gappy data. It can be run in parallel to speed up computation of the pseudocovariance and coefficients.

The method is iterated until there is either 1) a less than 1\% average change in imputed data for at least X iterations or 2) the maximum number of iterations is reached.

This package was developed to create a low-order representation of aircraft trajectories for generative modelling. If you use this package please cite the paper referenced below.

## Workflow

The package implements a fPCA algorithm with a pseudocovariance calculated with gappy data to replace a covariance matrix for the first step. PCA weights are fitted with a optimisation function. Subsequent steps use reconstructed data to calculate the full covariance.

INCLUDE WORKFLOW DIAGRAM HERE

## Installation

To install the package you can use `pip` after cloning. It is recommended to use a virtual environment to avoid conflicts.

	git clone https://github.com/amyhodgkin/gappyfpca.git
	cd gappyfpca
	pip install .

Dependencies required for included tests can be downloaded using

	pip install xxx
 
## Getting Started

To get started please see the `get_started` notebook which explains the main functions of the package and demonstrates these with a simple example. Considerations for the data format are also included here.

The package is suitable for small datasets. As an example, the algorithm takes 75 s per iteration to compute the principal components for a set of 5000 functions of length 100 on 16 cores.

 ## Citation

	@article{hodgkin2025probabilistic,
	  title={Probabilistic Simulation of Aircraft Descent via a Hybrid Physics-Data Approach},
	  author={Hodgkin, Amy and Pepper, Nick and Thomas, Marc},
	  journal={arXiv preprint arXiv:2504.02529},
	  year={2025}
	}
