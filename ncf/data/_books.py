import pandas as pd
import numpy as np
from ._dataset import Dataset
from pathlib import Path


def read_books() -> Dataset:
    """Load Books data into a Dataset.
    
    Returns
    -------
    Dataset
        Dataset object with loaded user-item ratings 
    """
    data_folder = Path.cwd() / Path("data")

    # Load only ratings, return Dataset
    cols = ["user_id", "book_id", "rating"]
    path = data_folder / "books" / "BX-Book-Ratings.csv"
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = cols

    # Get numpy arrays
    user_ids = df.user_id.astype(np.int32).values
    book_ids = df.book_id.astype(np.str_).values
    ratings = df.rating.astype(np.float32).values
    return Dataset(user_ids=user_ids, item_ids=book_ids, ratings=ratings)
