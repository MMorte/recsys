import torch
import torch.nn as nn
from typing import Any


def ifnone(a: Any, b: Any) -> Any:
    """'a' if 'a' is not None, otherwise 'b'.
    
    Parameters
    ----------
    a : Any
    b : Any
    
    Returns
    -------
    Any
        'a' or 'b' if a is None
    """
    return b if a is None else a


def trunc_normal_(x: torch.tensor, mean: float = 0.0, std: float = 1.0) -> torch.tensor:
    "Truncated normal initialization."
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


def embedding(ni: int, nf: int) -> nn.Module:
    "Create an embedding layer."
    emb = nn.Embedding(ni, nf)
    # See https://arxiv.org/abs/1711.09160
    with torch.no_grad():
        trunc_normal_(emb.weight, std=0.01)
    return emb
