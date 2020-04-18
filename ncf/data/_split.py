import numpy as np
from typing import Any, Tuple
from ._dataset import Dataset


def train_test_split(
    data: Dataset,
    shuffle: np.bool = True,
    test_size: np.float16 = 0.2,
    random_state: np.int_ or None = None,
) -> Tuple[Dataset, Dataset]:
    """Split dataset into random train and test subsets.
    
    Parameters
    ----------
    data : Dataset
        data we want to split
    shuffle : np.bool, optional
        whether or not to shuffle the data before splitting, by default True
    test_size : np.float16, optional
        represents the proportion of the dataset to include in the test split, by default 0.2
    random_state : np.int_ or None, optional
        random_state is the seed used by the random number generator, by default None
    
    Returns
    -------
    Tuple[Dataset, Dataset]
        train, test Dataset instances, split according to test_size percentage (and shuffled)
    """
    # Get indices as np.ndarray of dim (data shape,)
    indices = np.arange(len(data))
    # Shuffle data if True
    if shuffle:
        # Solve random state
        if random_state is None:
            random_state = np.random.RandomState()
        else:
            random_state = np.random.RandomState(random_state)
        random_state.shuffle(indices)

    # Obtain indices for train, test subsets
    cutoff = int((1.0 - test_size) * len(data))
    train_slice = slice(None, cutoff)
    test_slice = slice(cutoff, None)
    train_indices = indices[train_slice]
    test_indices = indices[test_slice]
    # Create train and test as a Dataset
    train = Dataset(
        user_ids=data.user_ids[train_indices],
        item_ids=data.item_ids[train_indices],
        ratings=data.ratings[train_indices],
        timestamps=data.timestamps[train_indices],
    )

    test = Dataset(
        user_ids=data.user_ids[test_indices],
        item_ids=data.item_ids[test_indices],
        ratings=data.ratings[test_indices],
        timestamps=data.timestamps[test_indices],
    )
    return train, test
