import numpy as np
import pytest

from gappyfpca.data_check import check_gappiness,clean_empty_data
from gappyfpca.eig import eig_decomp, fpca_num_coefs
from gappyfpca.fpca import reconstruct_func, l2_error
from gappyfpca.nancov import nancov
from gappyfpca.weights import fpca_weights

def test_check_gappiness():

    # Test with a valid dataset
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    try:
        result = check_gappiness(data)
        assert result is None
    except ValueError as e:
        assert False, f"Unexpected ValueError: {e}"

    # Test with a dataset containing all NaN values in a row
    data_with_nan_row = np.array([[1, 2, 3], [np.nan, np.nan, np.nan], [7, 8, 9]])
    with pytest.raises(ValueError, match="Rows"):
        check_gappiness(data_with_nan_row)

    # Test with a dataset containing all NaN values in a column
    data_with_nan_col = np.array([[1, 2, np.nan], [4, 5, np.nan], [7, 8, np.nan]])
    with pytest.raises(ValueError, match="Columns"):
        check_gappiness(data_with_nan_col)

    # Test with a dataset containing NaN values in the dot product
    data_with_nan_dot = np.array([[np.nan, 2, 3], [np.nan, 5, 6], [7, np.nan, 9]])
    with pytest.raises(ValueError, match="Dot of data contains NaN values"):
        check_gappiness(data_with_nan_dot) # Corrected variable used here

def test_clean_empty_data():
    # Test with a dataset that has empty rows and columns
    data = np.array([[1, 2, np.nan], [np.nan, np.nan, np.nan], [7, 8, np.nan]])    
    cleaned_data = clean_empty_data(data)
    
    # Check if the cleaned data contains the expected values
    expected_values = np.array([[1, 2], [7, 8]])
    assert np.array_equal(cleaned_data, expected_values)

    # Test with a dataset that has no empty rows or columns
    data_no_empty = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cleaned_data_no_empty = clean_empty_data(data_no_empty)
    
    assert np.array_equal(cleaned_data_no_empty, data_no_empty)


# do for iparallel=1 too
@pytest.mark.parametrize("iparallel", [0, 1])
def test_nancov1(iparallel):

    # Test it works out the correct covariance for complete data
    A=np.random.rand(3,3)

    nan_cov = nancov(A)

    cov = np.cov(A,bias=True,rowvar=False)

    #cov=np.dot(A.T,A)/(len(A[:,0]))

    check = np.isclose(nan_cov,cov).all()

    assert check

def test_nancov2():
    #test on fixed gappy data
    A = np.array([[1,np.nan,2],[np.nan, 2, 4],[0,1,np.nan]])
    cov=nancov(A)
    ans=np.array([[0.25,0.25,-0.5],[0.25, 0.25, 0.5],[-0.5,0.5,1]])
    assert np.isclose(cov, ans).all()

def test_eigdecomp():
    cov=np.array([[4, 0, 0],
                  [0, 9, 0],
                  [0, 0, 16]])

    eval,evec=eig_decomp(cov)

    eval_ans=np.array([ 16,9,4])
    evec_ans=np.array([[0, 0,  1],
                    [ 0,  1, 0],
                    [ 1, 0,  0]])
    
    check = np.isclose(eval, eval_ans).all() and np.isclose(evec, evec_ans).all()
    assert check
  
def test_fpca_num_coefs():

    evalue=[4,3,2,1]
    var_rat=0.9

    ncoefs=fpca_num_coefs(evalue,var_rat)

    assert ncoefs==3

@pytest.mark.parametrize("iparallel", [0, 1])
def test_fpca_weights1(iparallel):
    #checks function computes correct weights with no missing data

    # Example data (3 data points, 5 features)
    X = np.array([[1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7]])

    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)

    # Step : Get the principal components (eigenvectors)
    PCs = np.array([[ 4.47213595e-01,  8.94427191e-01, -4.29987528e-17],
       [ 4.47213595e-01, -2.23606798e-01,  8.66025404e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01]])

    # Step 4: Compute the weights (scores) for each data point
    ans_weights = np.dot(X_centered, PCs)

    weights=fpca_weights(X_centered.T,PCs,iparallel=iparallel)
    
    check=np.isclose(ans_weights,weights).all()

    assert check

def test_fpca_weights2():
    #check function runs with gappy data

    # Example data (3 data points, 5 features)
    X_gap = np.array([[1,np.nan, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, np.nan, np.nan]])
    
    X_gap_cent=X_gap-np.nanmean(X_gap,axis=0)

    # Step 3: Get the principal components (eigenvectors)
    PCs = np.array([[ 4.47213595e-01,  8.94427191e-01, -4.29987528e-17],
       [ 4.47213595e-01, -2.23606798e-01,  8.66025404e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01],
       [ 4.47213595e-01, -2.23606798e-01, -2.88675135e-01]])

    try:
        weights=fpca_weights(X_gap_cent.T,PCs)
        assert True
    except:
        assert False