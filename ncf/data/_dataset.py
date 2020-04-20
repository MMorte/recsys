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
        n_users: int = None,
        n_items: int = None,
        n_ratings: int = None,
    ):
        # Load dataset
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        self.timestamps = timestamps

        # Load dataset dimensions
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings

        # Load input (users, items) dimensions and map user/items to integers
        # Otherwise the dimensions (user count, item count, ...)
        # Which leads to inproper Embedding shapes
        if n_users is None or n_items is None:
            self.u_mapping = {
                old: new for new, old in enumerate(np.unique(self.user_ids))
            }
            self.i_mapping = {
                old: new for new, old in enumerate(np.unique(self.item_ids))
            }

            # New input ids, list-comprehension since theres no .map() for numpy
            # See: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
            self.user_ids = np.array([self.u_mapping[u_i] for u_i in self.user_ids])
            self.item_ids = np.array([self.i_mapping[i_i] for i_i in self.item_ids])

            # Load dataset information for model, __repr__, __len__, ...
            self.n_users = self.user_ids.max() + 1  # indexing from 0 thus:
            self.n_items = self.item_ids.max() + 1  # nn.Embedding(shape+1, n_factors)

        # Ratings loaded separately (due to different loading)
        if n_ratings is None:
            self.n_ratings = ratings.shape[0]

    def __repr__(self):
        representation = f"Data(users={self.n_users}, items={self.n_items}, interactions={self.n_ratings})"
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
        if self.timestamps is None:
            timestamps = None
        else:
            timestamps = torch.from_numpy(self.timestamps).to(device)
        return Dataset(
            user_ids=user_ids,
            item_ids=item_ids,
            ratings=ratings,
            timestamps=timestamps,
            n_users=self.n_users,
            n_items=self.n_items,
            n_ratings=self.n_ratings,
        )
