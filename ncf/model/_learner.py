import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam, SGD
from ncf.model._utils import ExponentialLR
from ncf.data import CollaborativeFilteringDataset
from ncf.model import EmbeddingNet
from ncf.visualization import Visualizer


class Learner:
    def __init__(
        self,
        data: CollaborativeFilteringDataset,
        model: torch.nn.Module = EmbeddingNet,
        criterion=torch.nn.MSELoss,
        optimizer: torch.optim.Optimizer = Adam,
        learning_rate: float = 1e-7,
        n_factors: int = 32,
        n_epochs: int = 5,
        batch_size: int = 64,
        y_range: Tuple[float, float] = None,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        random_state: int = None,
        num_workers: int = 2,
    ):
        # Data, model, loss function and optimizer
        self.data = data

        # Load hyperparameters
        self.learning_rate = learning_rate
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.y_range = y_range
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Initiate model, optimizer and loss function
        self._init_model(model=model, optimizer=optimizer, criterion=criterion)

        # Utilities
        self.random_state = random_state
        self.num_workers = num_workers

        # Obtain train, valid loaders
        self.train_loader, self.valid_loader = self._train_val_split()

        # Visualizations
        self.visualize = Visualizer(learner=self)

    def _init_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.MSELoss,
    ):
        """Data for the model initialization
        
        Parameters
        ----------
        model : torch.nn.Module
            Same as __init__
        optimizer : torch.optim.Optimizer
            Same as __init__
        criterion : torch.nn.MSELoss
            Same as __init__
        """
        # Same vars for lazy loading
        self.model_, self.optimizer_, self.criterion_ = (
            model,
            optimizer,
            criterion,
        )
        # Fetch device (gpu/cpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = model(
            n_factors=self.n_factors,
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            y_range=self.y_range,
        )
        # Move model to device
        self.model = self.model.to(self.device)

        # Criterion
        self.criterion = criterion()

        # Optimizer
        if optimizer == Adam:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif optimizer == SGD:
            self.optimizer = optimizer(
                params=self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
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
        diverge_th: float = 3,
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
            diverge threshold, if the epoch associated loss diverges from best loss we stop the finder, by default 3
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

        print(
            "Learning rate search finished. See the graph with {Learner}.visualize.lr_finder()"
        )

    def fit(self, learning_rate: Tuple[float, float]):
        # Capture learning errors
        self.train_val_error = {"train": [], "validation": [], "lr": []}
        self._init_model(
            model=self.model_, optimizer=self.optimizer_, criterion=self.criterion_
        )

        # Setup one cycle policy
        scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(self.train_loader),
            epochs=self.n_epochs,
            anneal_strategy="cos",
        )

        # Iterate over epochs
        for epoch in range(self.n_epochs):
            # Training set
            self.model.train()
            train_loss = 0
            for batch_num, samples in enumerate(self.train_loader):
                # Forward pass, get loss
                loss = self._forward_pass(samples=samples)
                train_loss += loss.item()

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update scheduler
                self.train_val_error["lr"].append(scheduler.get_lr()[0])
                # One cycle scheduler must be called per batch
                # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
                scheduler.step()

            # Append train loss per current epoch
            train_err = train_loss / batch_num
            self.train_val_error["train"].append(train_err)

            # Validation set
            self.model.eval()
            validation_loss = 0
            for batch_num, samples in enumerate(self.valid_loader):
                # Forward pass, get loss
                loss = self._forward_pass(samples=samples)
                validation_loss += loss.item()
            # Append validation loss per current epoch
            val_err = validation_loss / batch_num
            self.train_val_error["validation"].append(val_err)

        return pd.DataFrame(data={
            'Train error' : self.train_val_error['train'],
            'Validation error': self.train_val_error['validation']
        })

    def _forward_pass(self, samples: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        # Fetch tensors for users, items and ratings
        users, items, ratings = samples

        # Forward pass
        pred = self.model(users=users, items=items)
        loss = self.criterion(pred, ratings)
        return loss
