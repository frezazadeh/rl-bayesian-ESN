"""
Utility functions for environment resetting and reproducibility.
"""

import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reset_env(env, seed: int = None):
    """
    Reset the Gym environment, handling both old and new Gym APIs.

    Args:
        env: Gym environment.
        seed: Optional seed for environment reset.

    Returns:
        Observation after reset.
    """
    if seed is not None:
        out = env.reset(seed=seed)
    else:
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out
