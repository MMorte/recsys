import torch
import numpy as np
from typing import Tuple
from ._utils import Embedding
from ..data._dataset import Dataset
from ._utils import RMSELoss


class EmbeddingNet(torch.nn.Module):
    """Dot model for collaborative filtering
    Creates a simple model with Embedding weights and biases for n_users and n_items, with n_factors latent factors.
    Takes the dot product of the embeddings and adds the bias, then if y_range is specified, feed the result to a sigmoid rescaled to go from y_range[0] to y_range[1].
    Parameters
    ----------
    n_factors : int
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
        n_factors: int,
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
                (n_users, n_factors),
                (n_items, n_factors),
                (n_users, 1),
                (n_items, 1),
            ]
        ]

    def forward(self, users: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:
        # Create indices
        user_indices = users - 1
        item_indices = items - 1
        # User-item dot product
        dot = self.u_weight(user_indices) * self.i_weight(item_indices)
        # Add bias
        res = (
            dot.sum(1)
            + self.u_bias(user_indices).squeeze()
            + self.i_bias(item_indices).squeeze()
        )
        # Scale if y_range is specified
        if self.y_range is None:
            return res
        return (
            torch.sigmoid(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        )


class CollaborativeFiltering:
    def __init__(
        self,
        n_factors: int = 32,
        n_epochs: int = 30,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        model: torch.nn.Module = EmbeddingNet,
        loss: "loss_function" = RMSELoss,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        random_state: int or None = None,
        y_range: Tuple[float, float] = None,
    ):
        # Initialize params
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        # Fetch device (gpu/cpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialization
        self._initialized = False

    def _initialize(self, data: Dataset):
        """Allows repeated calls to fit.
        
        Parameters
        ----------
        data : Dataset
            Train data
        """
        # Model, Loss, optimizer
        self.model = self.model(
            n_factors=self.n_factors,
            n_users=data.n_users,
            n_items=data.n_items,
            y_range=(0.0, 5.0),
        ).to(self.device)
        self.criterion = self.loss()
        self.optimizer = self.optimizer(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self._initialized = True

    def fit(self, data: Dataset, verbose: np.bool = False):
        """Fit model to data.
        
        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------
        data : Dataset
            Training data 
        verbose : np.bool, optional
            Whether to print value of loss each Nth (5th) epoch, by default False
        """

        # Convert data to tensors
        data = data.to_tensor()
        # Get users, items, ratings
        users = data.user_ids
        items = data.item_ids
        ratings = data.ratings
        # Get input dimension for shuffling, mini-batching
        dim = len(data)
        # Initialize Model, Optimizer, Criterion
        if not self._initialized:
            self._initialize(data=data)
        # Begin training
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            # Get random permutation each epoch for shuffle
            # See: https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
            permutation = torch.randperm(dim)
            for index in range(0, dim, self.batch_size):
                # Obtain batch data
                indices = permutation[index : index + self.batch_size]
                batch_users, batch_items, batch_ratings = (
                    users[indices],
                    items[indices],
                    ratings[indices],
                )

                # Forward pass: Compute predicted y by passing x to the model
                pred = self.model(users=batch_users, items=batch_items)
                # Compute and add loss
                loss = self.criterion(pred, batch_ratings)
                epoch_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Scale loss and print
            n_batches = dim // self.batch_size + dim % self.batch_size
            self.current_loss = epoch_loss / n_batches
            if verbose:
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: loss {self.current_loss}")

    def evaluate(self, data: Dataset):
        """Calculate RMSELoss on the test set.
        
        Parameters
        ----------
        data : Dataset
            Test set data
        
        Returns
        -------
        torch.float
            Test RMSEloss 
        """
        # Convert data to tensors
        data = data.to_tensor()
        # Get users, items, ratings
        users = data.user_ids
        items = data.item_ids
        ratings = data.ratings
        # Forward pass to obtain predictions
        pred = self.model(users=users, items=items)
        # Calculate RMSELoss
        loss = self.criterion(pred, ratings)
        return loss.item()
