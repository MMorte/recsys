import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from typing_extensions import Literal, NoReturn
from typing import Tuple
from pathlib import Path


class RecommenderDataset(Dataset):
    """RecommenderDataset dataset
    
    Parameters
    ----------
    dataset : DATASET
        One of the supported datasets

    """

    DATASET = Literal["ml-100k", "books"]

    def __init__(self, dataset: DATASET):
        # Initialize dataset
        self.dataset = dataset
        self._init_dataset()

    def __len__(self):
        # as if n_ratings
        return len(self.df)

    def __repr__(self):
        representation = f"RecommenderDataset(users={self.n_users}, items={self.n_items}, interactions={len(self)})"
        return representation

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Get values from df
        users = self.df.loc[index, "user_id"]
        items = self.df.loc[index, "item_id"]
        ratings = self.df.loc[index, "rating"]
        return self._to_tensor(users, items, ratings)

    def _init_dataset(self) -> NoReturn:
        """Load dataframe with pd.read_csv to self.df
        
        Returns
        -------
        NoReturn
        """
        # Directory containing datasets
        data_dir = Path.cwd() / Path("data")

        # Currently support only explicit user-item ratings
        columns = ["user_id", "item_id", "rating"]

        # Read data
        # Loads the MovieLens data
        if self.dataset == "ml-100k":
            self.root_dir = data_dir / "ml-100k"
            csv_file = self.root_dir / "u.data"
            df = pd.read_csv(
                csv_file, sep="\t", names=columns, usecols=range(3), encoding="latin-1"
            )

        # Loads the 'Books' data
        elif self.dataset == "books":
            self.root_dir = data_dir / "books"
            csv_file = self.root_dir / "BX-Book-Ratings.csv"
            df = pd.read_csv(
                csv_file, sep=";", encoding="latin-1", skiprows=1, names=columns
            )

        # Create mapping for users, items
        self.user_mapping = {old: new for new, old in enumerate(df.user_id.unique())}
        self.item_mapping = {old: new for new, old in enumerate(df.item_id.unique())}

        # Map old:new values
        df.loc[:, "user_id"] = df.user_id.map(self.user_mapping)
        df.loc[:, "item_id"] = df.item_id.map(self.item_mapping)

        # Load counts for Embeddings and Forward prop
        self.n_users = df.user_id.max() + 1
        self.n_items = df.item_id.max() + 1
        self.df = df

    def _to_tensor(
        self,
        users: pd.Series or np.array or int,
        items: pd.Series or np.array or int,
        ratings: pd.Series or np.array or int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sample containing user_ids, item_ids and ratings  from __getitem__ to tensors.
        
        Parameters
        ----------
        users : pd.Seriesor np.array or int
            Values from self.df.user_id
        items : pd.Seriesor np.array or int
            Values from self.df.item_id
        ratings : pd.Seriesor np.array or int
            Values from self.df.ratings
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tensors of users, items, ratings
        """
        # Setup GPU if available, else use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert inputs and ratings to tensors
        users = torch.from_numpy(np.array(users)).to(torch.int64).to(device)
        items = torch.from_numpy(np.array(items)).to(torch.int64).to(device)
        ratings = torch.from_numpy(np.array(ratings)).float().to(device)
        return users, items, ratings
