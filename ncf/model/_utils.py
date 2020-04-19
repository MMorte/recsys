import torch
import torch.nn as nn
from typing import Any


def Embedding(ni: int, nf: int) -> nn.Embedding:
    """Create an embedding layer (shape ni x nf) using truncated normal initialization.
    See: https://arxiv.org/abs/1711.09160 [Embeddings]
         https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12 [Initialization]
    
    Parameters
    ----------
    ni : int
        Number of inputs
    nf : int
        Number of factors
    
    Returns
    -------
    nn.Embedding
        
    """
    # Embedding
    emb = nn.Embedding(ni, nf)
    # Initialization
    std = 0.01
    mean = 0
    with torch.no_grad():
        # as if "inplace=True"
        emb.weight.normal_().fmod_(2).mul_(std).add_(mean)
    return emb


class RMSELoss(nn.Module):
    """RMSELoss with eps to avoid tensor([nan]) during backprop.
    See: https://discuss.pytorch.org/t/rmse-loss-function/16540/3

    Parameters
    ----------
    eps : float
        Small num to avoid nan, defaults to 1e-6
    
    Returns
    -------
    loss : float
    """

    def __init__(self, eps=1e-20):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
