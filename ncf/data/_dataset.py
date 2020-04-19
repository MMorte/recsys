import torch
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
        self.n_users = user_ids.max()
        self.n_items = item_ids.max()
        self.n_ratings = ratings.shape[0]

    def __repr__(self):
        representation = f"Dataset contains {self.n_users} users, {self.n_items} items and {self.n_ratings} interactions."
        return representation

    def __len__(self):
        return self.n_ratings

    def to_tensor(self):
        """Convert input numpy arrays to tensors and setup proper device (GPU if available).
        """
        # Setup GPU if available, else use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert inputs and ratings to tensors
        user_ids = torch.from_numpy(self.user_ids).to(torch.int64).to(device)
        item_ids = torch.from_numpy(self.item_ids).to(torch.int64).to(device)
        ratings = torch.from_numpy(self.ratings).to(device)
        timestamps = torch.from_numpy(self.timestamps).to(device)
        return Dataset(
            user_ids=user_ids, item_ids=item_ids, ratings=ratings, timestamps=timestamps
        )
