import os, random
import numpy as np

def derive_seeds(master_seed: int, n: int = 4) -> list[int]:
    """Turn one master seed into n independent, high-quality sub-seeds."""
    ss = np.random.SeedSequence(master_seed)
    return [int(s) for s in ss.generate_state(n)]

def seed_everything(master_seed: int) -> None:
    """Set all *global* RNGs. Use for libraries that read global state."""
    os.environ["PYTHONHASHSEED"] = str(master_seed)
    random.seed(master_seed)
    np.random.seed(master_seed)
    try:
        import torch
        torch.manual_seed(master_seed)
        torch.cuda.manual_seed_all(master_seed)
        # Trade a little speed for reproducibility on GPU:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
