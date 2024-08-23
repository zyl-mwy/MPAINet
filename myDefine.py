import torch

import random
import math
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def rule(epoch):
     lamda = math.pow(0.1, epoch // 50)
     return lamda