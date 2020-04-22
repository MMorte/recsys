import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from recsys.data import RecommenderDataset


class Books:
    def __init__(self, root_dir: str = "data/books/"):
        # Root directory containing csv files
        self.root_dir = root_dir

        # Merged dataframe
        self.df = self._load_data()

        # User-item mappings from dataset
        self.user_mapping, self.item_mapping = self._load_mappings()

        # Load trained embeddings
        self.embeddings = self._load_embeddings()

    def _load_data(self) -> pd.DataFrame:
        """Load the BX-Books dataset into one merged dataframe.
        
        Returns
        -------
        pd.DataFrame
            BX-Books dataset containing ratings, books and users
        """
        # Data consists of 3 csv files
        # Ratings
        ratings_file = self.root_dir + "BX-Book-Ratings.csv"
        ratings_cols = ["user_id", "item_id", "rating"]
        ratings = pd.read_csv(
            ratings_file, sep=";", encoding="latin-1", skiprows=1, names=ratings_cols
        )

        # Books
        books_file = self.root_dir + "BX-Books.csv"
        books_cols = [
            "item_id",
            "title",
            "author",
            "year_published",
            "publisher",
            "images",
        ]
        books = pd.read_csv(
            books_file,
            sep=";",
            escapechar="\\",
            encoding="CP1252",
            skiprows=1,
            names=books_cols,
            usecols=range(6),
        )

        # Users
        users_file = self.root_dir + "BX-Users.csv"
        users_cols = ["user_id", "location", "age"]
        users = pd.read_csv(
            users_file,
            sep=";",
            escapechar="\\",
            encoding="CP1252",
            skiprows=1,
            names=users_cols,
            usecols=range(3),
        )

        # Merge together
        df = pd.merge(ratings, users, on="user_id", how="left")
        df = pd.merge(df, books, on="item_id", how="left")
        return df

    def _load_mappings(self) -> Tuple[dict, dict]:
        """Load original:new user and item ID mappings
        
        Returns
        -------
        Tuple[dict, dict]
            user and item id mappings
        """
        data = RecommenderDataset(dataset="books")
        u_mapping = data.user_mapping
        i_mapping = data.item_mapping
        return u_mapping, i_mapping

    def _load_embeddings(self) -> dict:
        """Load pretrained embeddings
        
        Returns
        -------
        OrderedDict
            Loaded pretrained embeddings
        """
        # Load saved state dict
        emb_file = self.root_dir + "embeddings_books.pt"
        emb = torch.load(emb_file, map_location=torch.device("cpu"))
        return emb

    def _fetch_lotr_users(self) -> torch.Tensor:
        """Fetch LotR users (based on their ratings).
        
        Returns
        -------
        torch.Tensor
            IDs of users who like Lord of the Rings
        """
        # Title contains LotR
        contains_lotr = self.df.loc[:, "title"].str.contains("Lord of the Rings")
        contains_lotr = contains_lotr.fillna(False)

        # Ratings above 8
        ratings_10 = self.df.loc[:, "rating"] > 8

        # Dataframe subset
        lotr_df = self.df.loc[contains_lotr & ratings_10, :]

        # Number of ratings
        users = lotr_df.groupby("user_id")["rating"].count()
        # Sorted by highest
        users = users.sort_values(ascending=False).index

        # Map to index
        users = users.map(self.user_mapping).values

        # Conver to tensor
        users = torch.from_numpy(users)
        return users

    def _fetch_all_books(self) -> torch.Tensor:
        """Get IDS of all books
        
        Returns
        -------
        torch.Tensor
            Book IDS
        """
        return torch.tensor(list(self.item_mapping.values()))

    def recommend(self, top_n: int = 5) -> torch.Tensor:
        """Recommend movies for user(s)
        
        Parameters
        ----------
        top_n : int
            How many books should we recommend
        
        Returns
        -------
        torch.Tensor
            Predicted book title(s)
        """
        # Load users to predict
        users = self._fetch_lotr_users()
        items = self._fetch_all_books()
        n_users = len(users)
        n_items = len(items)

        # Load embeddings and y_range for scaling
        y_range = (0.0, 10.0)
        u_weight = self.embeddings["u_weight.weight"]
        i_weight = self.embeddings["i_weight.weight"]
        u_bias = self.embeddings["u_bias.weight"]
        i_bias = self.embeddings["i_bias.weight"]

        # Shuffle indices
        np.random.shuffle(items)

        # Save predictions
        predictions = {"items": [], "preds": []}

        # Iterate over books and give ratings
        for i in range(0, n_items, n_users):
            items_i = items[i : i + n_users]

            # Trim users for last iteration so tensor shapes ==
            if len(users) != len(items_i):
                users = users[:len(items_i)]

            # Make predictions
            dot = u_weight[users] * i_weight[items_i]
            # Add bias
            res = dot.sum(1) + u_bias[users].squeeze() + i_bias[items_i].squeeze()
            # Scale if y_range
            pred = torch.sigmoid(res) * (y_range[1] - y_range[0]) + y_range[0]

            # Save
            predictions["items"].append(items_i)
            predictions["preds"].append(pred)
        return self._pred2rec(predictions=predictions, top_n=top_n)

    def _pred2rec(self, predictions: dict, top_n: int = 5) -> pd.DataFrame:
        """Convert predictions dictionary to Top N titles

        Parameters
        ----------
        predictions : dict
            Dict from self.recommend()
        top_n : int, optional
            How many recs, by default 5

        Returns
        -------
        pd.DataFrame
            Formatted recommendations
        """        
        # Unpack
        predictions["items"] = [
            item.item() for sublist in predictions["items"] for item in sublist
        ]
        predictions["preds"] = [
            item.item() for sublist in predictions["preds"] for item in sublist
        ]

        # Pandas transformations for a prettier output
        recommendations = pd.DataFrame(data=predictions)
        recommendations = recommendations.sort_values(by='preds', ascending=False)
        recommended_books_ids = recommendations.loc[:, 'items'].head(top_n).values
        recommended_books = self.df.item_id.map(self.item_mapping).isin(recommended_books_ids)
        top_n_recs = self.df.loc[recommended_books, 'title'].unique()
        recommendations = pd.DataFrame(data={f'Top {top_n} Recommendations':top_n_recs})
        return recommendations
