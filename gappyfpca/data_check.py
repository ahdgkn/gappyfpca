
import numpy as np

def check_gappiness(data):

    # Dot product of the data matrix must be that no NaN values are present
    nan_mask = ~np.isnan(data)
    # first check if any rows or columns are fully empty
    if np.any(np.sum(nan_mask, axis=0) == 0):
        raise ValueError("Columns",np.where(np.sum(nan_mask,axis=0)==0)  ,"have all NaN values. Please remove them before proceeding.")
    if np.any(np.sum(nan_mask, axis=1) == 0):
        raise ValueError("Rows",np.where(np.sum(nan_mask,axis=1)==0) ,"have all NaN values. Please remove them before proceeding.")
    # Check if the dot product of the data matrix contains NaN values
    N = np.dot(nan_mask.astype(int).T, nan_mask.astype(int))
    print(N)
    if np.any(N == 0):
        offending_indices = np.where(N == 0)
        offending_pairs = list(zip(offending_indices[0], offending_indices[1]))
        raise ValueError(
            f"Dot of data contains NaN values. Offending row-column combinations are: {offending_pairs}. "
            "Please handle them before proceeding."
        )

    else:
        print("Data is suitable for gappy fPCA method")

    return None