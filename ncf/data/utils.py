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
    """

    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray):
        # Load dataset
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        # Load dataset information for __repr__, __len__
        self.num_users = np.unique(user_ids).shape[0]
        self.num_items = np.unique(item_ids).shape[0]
        self.num_ratings = ratings.shape[0]

    def __repr__(self):
        representation = f"Dataset contains {self.num_users} users, {self.num_items} items and {self.num_ratings} interactions."
        return representation

    def __len__(self):
        return self.num_ratings
