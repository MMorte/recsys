import numpy as np


class Dataset:
    """Dataset class for explicit feedback data. Must contain users, items and ratings. 
    
    Parameters
    ----------
    user_ids : np.ndarray
        np.int32, array/column containing user ids
    item_ids : np.ndarray
        np.int32, array/column containing item ids
    ratings : np.ndarray
        np.float32, array/column containing ratings
    timestamps : np.ndarray, optional
        np.int64, array/column containing timestamps
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: np.ndarray,
        timestamps: np.ndarray = None,
    ):
        # Load dataset
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps
        # Load dataset information for __repr__, __len__
        self.n_users = np.unique(user_ids).shape[0]
        self.n_items = np.unique(item_ids).shape[0]
        self.n_ratings = ratings.shape[0]

    def __repr__(self):
        representation = f"Dataset contains {self.n_users} users, {self.n_items} items and {self.n_ratings} interactions."
        return representation

    def __len__(self):
        return self.n_ratings
