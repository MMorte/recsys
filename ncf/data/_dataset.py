import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Literal
from pathlib import Path


class CollaborativeFilteringDataset(Dataset):
    "Collabarive Filtering dataset"
    DATASET = Literal["ml-100k", "books"]
       
    def __init__(self, dataset: DATASET):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self._init_dataset()

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        representation = f"CollaborativeFilteringDataset(users={self.df.user_id.nunique()}, items={self.df.item_id.nunique()}, interactions={len(self)})"
        return representation
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # Get values from df
        users = self.df.loc[index, 'user_id']
        items = self.df.loc[index, 'item_id']
        ratings = self.df.loc[index, 'rating']
        return self._to_tensor(users, items, ratings)

    def _init_dataset(self):
        data_dir = Path.cwd() / Path("data")
        # Currently support only explicit user-item ratings
        columns = ["user_id", "item_id", "rating"]
        if self.dataset == "ml-100k":
            # Loads the MovieLens data
            self.root_dir = data_dir / "ml-100k"
            csv_file = self.root_dir / "u.data"
            self.df = pd.read_csv(
                csv_file, sep="\t", names=columns, usecols=range(3), encoding="latin-1"
            )
        elif self.dataset == "books":
            # Loads the 'Books' data
            self.root_dir = data_dir / "books"
            csv_file = self.root_dir / "BX-Book-Ratings.csv"
            df = pd.read_csv(
                csv_file, sep=";", encoding="latin-1", skiprows=1, names=columns
            )
            # Create mapping for users, items
            self.user_mapping = {old: new for new, old in enumerate(df.user_id.unique())}
            self.item_mapping = {old: new for new, old in enumerate(df.item_id.unique())}

            # New values
            df.loc[:, 'user_id'] = df.user_id.map(self.user_mapping)
            df.loc[:, 'item_id'] = df.item_id.map(self.item_mapping)
            self.df = df
                
    def _to_tensor(self, users, items, ratings):
        # Setup GPU if available, else use CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Convert inputs and ratings to tensors
        users = torch.from_numpy(np.array(users)).to(torch.int64).to(device)
        items = torch.from_numpy(np.array(items)).to(torch.int64).to(device)
        ratings = torch.from_numpy(np.array(ratings)).to(torch.float64).to(device)
        return users, items, ratings
