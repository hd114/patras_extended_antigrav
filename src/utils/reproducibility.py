import random
import os
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    # Python random
    random.seed(seed)
    
    # Environment variables for hashing (important for some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # PyTorch Deterministic backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to: {seed}")
