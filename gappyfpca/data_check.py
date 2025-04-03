
import numpy as np


def check_gappiness(data: np.ndarray) -> None:

    # Dot product of the data matrix must be that no NaN values are present
    nan_mask = ~np.isnan(data)
    # first check if any rows or columns are fully empty
    if np.any(np.sum(nan_mask, axis=0) == 0):
        raise ValueError("Columns",np.where(np.sum(nan_mask,axis=0)==0)  ,"have all NaN values. Please remove them before proceeding.")
    if np.any(np.sum(nan_mask, axis=1) == 0):
        raise ValueError("Rows",np.where(np.sum(nan_mask,axis=1)==0) ,"have all NaN values. Please remove them before proceeding.")
    # Check if the dot product of the data matrix contains NaN values
    N = np.dot(nan_mask.astype(int).T, nan_mask.astype(int))

    if np.any(N == 0):
        offending_indices = np.where(N == 0)
        offending_pairs = list(zip(offending_indices[0], offending_indices[1], strict=False))
        raise ValueError(
            f"Dot of data contains NaN values. Offending row-column combinations are: {offending_pairs}. "
            "Please handle them before proceeding."
        )

    print("Data is suitable for gappy fPCA method")

    return

def clean_empty_data(data:np.ndarray) -> np.ndarray:
    """
    Cleans the data by removing rows and columns that are fully empty (NaN).
    """
    # Remove rows and columns that are fully empty
    data_cleaned = data[~np.isnan(data).all(axis=1), :]
    data_cleaned[:, ~np.isnan(data_cleaned).all(axis=0)]
    print(len(data)-len(data_cleaned),"rows and",data.shape[1]-data_cleaned.shape[1],"columns were removed from the data.")
    return data_cleaned


def remove_toomuch_gappiness(data:np.ndarray, min_data: int =2) -> np.ndarray:
    """
    Removes gappiest rows until the data pseudocovariance can be computed.
    """

    # check data already has enough rows
    if data.shape[0] < min_data:
        raise ValueError("Data array only contains",data.shape[0],"measurements with a supplied minimum of",min_data)

    nan_mask = ~np.isnan(data)
    data_cleaned = np.copy(data)

    N = np.dot(nan_mask.astype(int).T, nan_mask.astype(int))

    if not np.any(N == 0):
        print('Data is suitable for gappy fPCA method')
        return data

    original_indices = np.arange(data.shape[0])
    removed_indices =[]

    while np.any(N == 0):

        nan_mask = ~np.isnan(data_cleaned)

        N = np.dot(nan_mask.astype(int).T, nan_mask.astype(int))
        offending_indices = np.where(N == 0)
        # Find unique columns involved in conflicts
        problem_cols = np.unique(np.concatenate(offending_indices))

        # Score rows based on how many conflicts they might be involved in
        row_scores = np.zeros(data_cleaned.shape[0])
        for r_idx in range(data_cleaned.shape[0]):
            # Check which problem columns have NaN in this row
            nan_in_problem_cols = np.isnan(data_cleaned[r_idx, problem_cols])
            # Simple score: count how many problem columns have NaN in this row
            row_scores[r_idx] = np.sum(nan_in_problem_cols)
            # Could also add a factor for total NaNs in the row

        # Find the row with the highest score (most involved in potential conflicts)
        row_to_remove_idx = np.argmax(row_scores)

        #remove the gappiest row first
        data_cleaned = np.delete(data_cleaned, row_to_remove_idx, axis=0)

    return data_cleaned