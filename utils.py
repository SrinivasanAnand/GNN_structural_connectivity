# Author: Anwar Said, https://github.com/Anwar-Said/NeuroGraph
# Description:
# Utils for deep learning experiments

import torch
import random
import numpy as np

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False