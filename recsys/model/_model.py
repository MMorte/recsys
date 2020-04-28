import torch
import logging
import numpy as np
from typing import Tuple
from ._utils import Embedding
from ..data._dataset import Dataset


class EmbeddingNet(torch.nn.Module):
    """Dot model for collaborative filtering
    Creates a simple model with Embedding weights and biases for n_users and n_items, with emb_size latent factors.
    Takes the dot product of the embeddings and adds the bias, then if y_range is specified, feed the result to a sigmoid rescaled to go from y_range[0] to y_range[1].
    Parameters
    ----------
    emb_size : int
        Number of factors (embedding size)
    n_users : int
        Number of users
    n_items : int
        Number of items
    y_range : Tuple[float, float], optional
        Scale for output e.g. y_range=(0., 5.) - ratings are usually on the scale from 1 to 5 (10, 100, ...), by default None
    """

    def __init__(
        self,
        emb_size: int,
        n_users: int,
        n_items: int,
        y_range: Tuple[float, float] = None,
    ):
        super().__init__()
        self.y_range = y_range
        # Get embeddings for users, items, get biases
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [
            Embedding(*o)
            for o in [
                (n_users, emb_size),
                (n_items, emb_size),
                (n_users, 1),
                (n_items, 1),
            ]
        ]

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # User-item dot product
        dot = self.u_weight(users) * self.i_weight(items)
        # Add bias
        result = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        # Scale if y_range is specified
        if self.y_range is None:
            return result
        return (
            torch.sigmoid(result) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        )
