from gappyfpca.fpca import *
from sklearn.decomposition import PCA

import numpy as np

import pytest


def test_nancov1():

    # Test it works out the correct covariance for complete data
    A=np.random.rand(3,3)

    nan_cov = nancov(A,iparallel=0)

    print(nan_cov)
    cov = np.cov(A,bias=True,rowvar=False)
    print(cov)
    #cov=np.dot(A.T,A)/(len(A[:,0]))

    check = np.isclose(nan_cov,cov).all()

    assert check

def test_nancov2():
    #test on fixed gappy data
    A = np.array([[1,np.nan,2],[np.nan, 2, 4],[0,1,np.nan]])
    cov=nancov(A,iparallel=0)
    ans=np.array([[0.25,0.25,-0.5],[0.25, 0.25, 0.5],[-0.5,0.5,1]])
    assert (cov==ans).all()

def test_eigsort():
    cov=np.array([[4, 0, 0],
                  [0, 9, 0],
                  [0, 0, 16]])

    eval,evec=find_and_sort_eig(cov)

    print(eval,evec)
    eval_ans=np.array([ 16,9,4])
    evec_ans=np.array([[0, 0,  1],
                    [ 0,  1, 0],
                    [ 1, 0,  0]])
    
    check = (eval==eval_ans).all() and (evec==evec_ans).all()
    assert check
  
def test_fpca_num_coefs():

    evalue=[4,3,2,1]
    var_rat=0.9

    ncoefs=fpca_num_coefs(evalue,var_rat)

    assert ncoefs==3

def test_fpca_weights1():
    #checks function computes correct weights with no missing data

    # Example data (3 data points, 5 features)
    X = np.array([[1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7]])

    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Perform PCA
    pca = PCA(n_components=3)
    pca.fit(X_centered)

    # Step 3: Get the principal components (eigenvectors)
    PCs = pca.components_.T

    # Step 4: Compute the weights (scores) for each data point
    ans_weights = np.dot(X_centered, PCs)

    weights=fpca_weights(X_centered.T,PCs)
    
    check=np.isclose(ans_weights,weights).all()

    assert check

def test_fpca_weights2():
    #check function runs with gappy data

    # Example data (3 data points, 5 features)
    X_gap = np.array([[1,np.nan, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, np.nan, np.nan]])
    
    X = np.array([[1,2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7]])

    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    X_gap_cent=X_gap-np.nanmean(X_gap,axis=0)

    # Step 2: Perform PCA
    pca = PCA(n_components=3)
    pca.fit(X_centered)

    # Step 3: Get the principal components (eigenvectors)
    PCs = pca.components_.T

    # Step 4: Compute the weights (scores) for each data point
    ans_weights = np.dot(X_centered, PCs)
    try:
        weights=fpca_weights(X_gap_cent.T,PCs)
        assert True

    except:
        assert False

def test_nancov_para():

    # Test it works out the correct covariance for complete data
    A=np.random.rand(3,3)

    nan_cov = nancov(A,iparallel=1)

    print(nan_cov)
    cov = np.cov(A,bias=True,rowvar=False)
    print(cov)
    #cov=np.dot(A.T,A)/(len(A[:,0]))

    check = np.isclose(nan_cov,cov).all()

    assert check

def test_fpca_weights_para():
    #checks function computes correct weights with no missing data

    # Example data (3 data points, 5 features)
    X = np.array([[1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7]])

    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Perform PCA
    pca = PCA(n_components=3)
    pca.fit(X_centered)

    # Step 3: Get the principal components (eigenvectors)
    PCs = pca.components_.T

    # Step 4: Compute the weights (scores) for each data point
    ans_weights = np.dot(X_centered, PCs)

    weights=fpca_weights(X_centered.T,PCs,iparallel=1)
    
    check=np.isclose(ans_weights,weights).all()

    assert check

def test_gappyfpca():
    """integration test for accuracy? of everything together """

    # generate synthetic dataset to test

    # Parameters
    M = 1000  # Number of functions
    L = 50   # Length of each function

    # Sinusoidal patterns with random frequencies and phases
    x = np.linspace(0, 2 * np.pi, L)
    functions = np.array([10+np.random.uniform(0.1, 5)*np.sin(x * np.random.uniform(1, 1.5) + np.random.uniform(0,  np.pi/2)) 
                      for _ in range(M)])

    # Random polynomials
    #functions = np.array([np.polyval(np.random.uniform(-1, 1, size=3), np.linspace(-1, 1, L)) 
    #                      for _ in range(M)])

    data=np.copy(functions)
    #artifically gappy it
    for i in range(M):
        # Determine the number of NaNs to insert (0 to <50% of the function length)
        num_nans = np.random.randint(0, L // 2)  
        # Randomly select indices to replace with NaN
        nan_indices = np.random.choice(L, num_nans, replace=False)
        # Replace selected indices with NaN
        data[i, nan_indices] = np.nan


    # Generate fpca of full data using
    fpca_comps,fpca_coefs,evalue,run_stat=gappyfpca(data,1,max_iter=15,num_iter=5,iparallel=0) # can i test parallel too?

    # Impute missing data

    function_recon=reconstruct_func(fpca_comps[0,:],fpca_comps[1:,:],fpca_coefs)

    mean_error = (np.mean(np.abs(functions-function_recon)))
    print(mean_error)
    if mean_error>=0.1:
        assert False

    else:
        assert True

