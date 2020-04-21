import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler
from ncf.model._utils import ExponentialLR
from ncf.data import CollaborativeFilteringDataset
from ncf.model import EmbeddingNet


class Learner:
    def __init__(
        self,
        data: CollaborativeFilteringDataset,
        model: torch.nn.Module = EmbeddingNet,
        criterion=torch.nn.MSELoss,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        learning_rate: float = 1e-7,
        n_factors: int = 32,
        n_epochs: int = 5,
        batch_size: int = 64,
        y_range: Tuple[float, float] = None,
        weight_decay: float = 0,
        random_state: int = None,
        num_workers: int = 2,
    ):
        # Data, model, loss function and optimizer
        self.data = data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Load hyperparameters
        self.learning_rate = learning_rate
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.y_range = y_range
        self.weight_decay = weight_decay

        # Utilities
        self.random_state = random_state
        self.num_workers = num_workers

        # Initiate model, optimizer and loss function
        self._init_model()

        # Obtain train, valid loaders
        self.train_loader, self.valid_loader = self._train_val_split()

    def _init_model(self):
        """Initialize 
        
        Parameters
        ----------
        data : Dataset
            Data for the model initialization
        """
        # Fetch device (gpu/cpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = self.model(
            n_factors=self.n_factors,
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            y_range=self.y_range,
        )
        # Move model to device
        self.model = self.model.to(self.device)

        # Criterion
        self.criterion = self.criterion()

        # Optimizer
        self.optimizer = self.optimizer(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _train_val_split(
        self, valid_size: float = 0.1, shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Utility function for loading and returning train and valid DataLoaders.
        
        Parameters
        ----------
        valid_size : float, optional
            Size of the validation set, by default 0.1
        shuffle : bool, optional
            Whether to shuffle the dataset or not, by default True
        
        Returns
        -------
        Tuple[DataLoader, DataLoader]
            Iterale train and valid dataloaders
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (valid_size >= 0) and (valid_size <= 1), error_msg

        # Create indices and a percentage split
        num_data = len(self.data)
        indices = list(range(num_data))
        split = int(np.floor(valid_size * num_data))

        # Shuffle the data
        if shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        # Create a sampler for the dataloader
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Create dataloaders from data with batch size and number of workers from __init__
        train_loader = torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            self.data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )
        return train_loader, valid_loader

    def lr_find(
        self,
        num_iter: int = 100,
        end_lr: float = 10,
        smooth_f: float = 0,
        diverge_th: float = 5,
    ):
        """Learning rate range test.
        The learning rate range test increases the learning rate in a pre-training run
        between two boundaries in an exponential manner. It provides valuable
        information on how well the network can be trained over a range of learning rates
        and what is the optimal learning rate.
        
        Parameters
        ----------
        num_iter : int, optional
            Number of iterations over which the test occurs, by default 100
        end_lr : float, optional
            Maximum learning rate to test, by default 10
        smooth_f : float, optional
            Loss smoothing factor within the [0, 1] interval. Disabled if set to 0, otherwise the loss is smoothed using exponential smoothing, by default 0
        diverge_th : float, optional
            [description], by default 5
        """
        # Test results
        self.history = {"lr": [], "loss": []}
        best_loss = None

        # Setup Scheduler
        scheduler = ExponentialLR(
            optimizer=self.optimizer, end_lr=end_lr, num_iter=num_iter
        )

        # Iterate over epochs
        for epoch in range(num_iter):

            # Iterate over batches
            epoch_loss = 0
            for batch_num, samples in enumerate(self.train_loader):
                # Fetch tensors for users, items and ratings
                users, items, ratings = samples

                # Forward pass
                pred = self.model(users=users, items=items)
                loss = self.criterion(pred, ratings)
                epoch_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            scheduler.step()
            self.history["lr"].append(scheduler.get_lr()[0])

            # Divide by count of batches for correct loss
            epoch_loss /= batch_num
            print(f"Epoch: {epoch} loss: {epoch_loss}")

            # Track the best loss and smooth it if smooth_f is specified
            if epoch == 0:
                best_loss = epoch_loss
            else:
                if smooth_f > 0:
                    epoch_loss = (
                        smooth_f * epoch_loss
                        + (1 - smooth_f) * self.history["loss"][-1]
                    )
                if epoch_loss < best_loss:
                    best_loss = epoch_loss

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(epoch_loss)
            if epoch_loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        print("Learning rate search finished. See the graph with {finder_name}.plot()")

    def plot(self, skip_start: int = 10, skip_end: int = 2):
        """Plots the learning rate range test.
        
        Parameters
        ----------
        skip_start : int, optional
            Number of batches to trim from the start, by default 10
        skip_end : int, optional
            Number of batches to trim from the start, by default 2
        
        Returns
        -------
        matplotlib.axes.Axes
            Object that contains the plot
        """
        # Load data for plotting
        lrs = self.history["lr"]
        losses = self.history["loss"]

        # Trim plot
        if skip_start > 0:
            lrs, losses = lrs[skip_start:], losses[skip_start:]
        if skip_end > 0:
            lrs, losses = lrs[:-skip_end], losses[:-skip_end]

        # Create plot
        fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Loss")
        if fig is not None:
            plt.show()
        return ax
