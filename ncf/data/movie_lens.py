import pandas as pd
import numpy as np
from ncf.data.utils import Dataset
from pathlib import Path

VARIANTS = ["100k", "1m", "10m", "20m"]
BASE_URL = "https://grouplens.org/datasets/movielens/"


def read_movie_lens(variant: str) -> Dataset:
    """Load MovieLens into a Dataset.
    
    Args:
        variant (str): variant of the MovieLens dataset (100k, 1m, ...)
    
    Returns:
        pd.DataFrame: Dataframe containing ratings, users and movies
    """
    data_folder = Path.cwd() / Path("data")
    if variant == "ml-100k":
        # Load only ratings, return Dataset
        cols = ["user_id", "movie_id", "rating", "timestamp"]
        path = data_folder / "ml-100k" / "u.data"
        df = pd.read_csv(path, sep="\t", names=cols, encoding="latin-1")

        # Get numpy arrays
        user_ids = df.user_id.astype(np.int32).values
        movie_ids = df.movie_id.astype(np.int32).values
        ratings = df.rating.astype(np.float32).values
        timestamps = df.timestamp.astype(np.int64).values
        return Dataset(
            user_ids=user_ids,
            item_ids=movie_ids,
            ratings=ratings,
            timestamps=timestamps,
        )
