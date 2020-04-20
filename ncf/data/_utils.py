import numpy as np


def array_or_none(array: np.ndarray or None, indexer: np.ndarray) -> np.ndarray:
    """Index array if array is not None
    
    Parameters
    ----------
    array : np.ndarray or None
        array we wont to index
    b : np.ndarray
        
    Returns
    -------
    np.ndarray
        a or a[indexer]
    """
    return array if array is None else array[indexer]
