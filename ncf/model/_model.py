import torch
import numpy as np
from typing import Tuple
from ._utils import embedding
from ..data._dataset import Dataset


class EmbeddingNet(torch.nn.Module):
    """Dot model for collaborative filtering
    Creates a simple model with Embedding weights and biases for n_users and n_items, with n_factors latent factors.
    Takes the dot product of the embeddings and adds the bias, then if y_range is specified, feed the result to a sigmoid rescaled to go from y_range[0] to y_range[1].
    Parameters
    ----------
    n_factors : int
        number of factors (embedding size)
    n_users : int
        number of users
    n_items : int
        number of items
    y_range : Tuple[float, float], optional
        scale for output e.g. y_range=(0., 5.) - ratings are usually on the scale from 1 to 5 (10, 100, ...), by default None
    """

    def __init__(
        self,
        n_factors: int,
        n_users: int,
        n_items: int,
        y_range: Tuple[float, float] = None,
    ):
        super().__init__()
        self.y_range = y_range
        # Get embeddings for users, items, get biases
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [
            embedding(*o)
            for o in [
                (n_users, n_factors),
                (n_items, n_factors),
                (n_users, 1),
                (n_items, 1),
            ]
        ]

    def forward(self, users: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:
        # User-item dot product
        dot = self.u_weight(users) * self.i_weight(items)
        # Add bias
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        # Scale if y_range is specified
        if self.y_range is None:
            return res
        return (
            torch.sigmoid(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        )


class CollaborativeFiltering:
    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 100,
        learning_rate: float = 1e-2,
        batch_size: int = 128,
        weight_decay: float = 0.1,
        model: torch.nn.Module = EmbeddingNet,
        loss: "loss_function" = torch.nn.MSELoss,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        random_state: int or None = None,
        y_range: Tuple[float, float] = None,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.use_cuda = torch.cuda.is_available()

    def fit(self, data: Dataset, verbose: np.bool = False):

        # Get users, items, ratings
        users = torch.from_numpy(data.user_ids).to(torch.int64)
        items = torch.from_numpy(data.item_ids).to(torch.int64)
        ratings = torch.from_numpy(data.ratings)
        # Get input dimension for shuffling, mini-batching
        dim = len(data)
        # Model, Loss, optimizer
        model = self.model(
            n_factors=self.n_factors,
            n_users=data.n_users,
            n_items=data.n_items,
            y_range=(0.0, 5.0),
        )
        criterion = self.loss(reduction="sum")
        optimizer = self.optimizer(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        for epoch in range(self.n_epochs):
            # Print loss every 10th epoch
            epoch_loss = 0.0
            # Get random permutation each epoch for shuffle
            # See: https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
            permutation = torch.randperm(dim)
            for i in range(0, dim, self.batch_size):
                # Obtain batch data
                indices = permutation[i : i + self.batch_size]
                batch_users, batch_items, batch_ratings = (
                    users[indices],
                    items[indices],
                    ratings[indices],
                )

                # Forward pass: Compute predicted y by passing x to the model
                pred = model(users=batch_users, items=batch_items)

                # Compute and add loss
                loss = criterion(pred, batch_ratings)
                epoch_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Scale loss and print
            epoch_loss /= i + 1
            if verbose:
                print("Epoch {}: loss {}".format(epoch, epoch_loss))
