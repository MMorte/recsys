import matplotlib.pyplot as plt

from ncf.model import Learner


class Visualizer:
    def __init__(self, learner: Learner):
        self.learner = learner

    def plot_lr_finder(self, skip_start: int = 10, skip_end: int = 2):
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
        lrs = self.learner.history["lr"]
        losses = self.learner.history["loss"]

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

    def plot_lr(self):
        """Plot the learning rate of a trained model as a function of number of iterations.
        
        Returns
        -------
        matplotlib.axes.Axes
            Object that contains the plot
        """
        lrs = self.learner.train_val_error["lr"]
        iters = [i for i in range(len(lrs))]

        # Create plot
        fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(iters, lrs)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Learning rate")
        if fig is not None:
            plt.show()
        return ax

    def plot_loss(self):
        """Plot train and validation loss as a function of number of epochs.

        Returns
        -------
        matplotlib.axes.Axes
            Object that contains the plot
        """
        train = self.learner.train_val_error["train"]
        validation = self.learner.train_val_error["validation"]
        iters = [i + 1 for i in range(len(train))]

        # Create plot
        fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(iters, train, label="train", color="blue")
        ax.plot(iters, validation, label="validation", color="orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if fig is not None:
            plt.show()
        return ax
