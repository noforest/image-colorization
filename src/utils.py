import random
import numpy as np
import torch

SEED = 42

def seed_everything(seed: int = SEED, deterministic: bool = False) -> None:
    """Fixe toutes les sources d'aléatoire du programme."""
    random.seed(seed)                 # module random de Python
    np.random.seed(seed)              # numpy (API globale)
    torch.manual_seed(seed)           # torch CPU + toutes les GPU

    if deterministic:                 # reproductibilité exacte sur GPU, mais plus lent
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False