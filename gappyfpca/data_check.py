import numpy as np


def check_gappiness(data: np.ndarray) -> None:
    # Dot product of the data matrix must be that no NaN values are present
    nan_mask = ~np.isnan(data)
    # first check if any rows or columns are fully empty
    if np.any(np.sum(nan_mask, axis=0) == 0):
        raise ValueError(
            "Columns",
            np.where(np.sum(nan_mask, axis=0) == 0),
            "have all NaN values. Please remove them before proceeding.",
        )
    if np.any(np.sum(nan_mask, axis=1) == 0):
        raise ValueError(
            "Rows",
            np.where(np.sum(nan_mask, axis=1) == 0),
            "have all NaN values. Please remove them before proceeding.",
        )
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


def clean_empty_data(data: np.ndarray) -> np.ndarray:
    """
    Cleans the data by removing rows and columns that are fully empty (NaN).
    """
    # Remove rows and columns that are fully empty
    data_cleaned = data[~np.isnan(data).all(axis=1), :]
    data_cleaned[:, ~np.isnan(data_cleaned).all(axis=0)]
    print(
        len(data) - len(data_cleaned),
        "rows and",
        data.shape[1] - data_cleaned.shape[1],
        "columns were removed from the data.",
    )
    return data_cleaned
