
import torch
import torch.nn as nn
import torch.distributions as td

import lightning.pytorch as pl

class DeepSymbolicOptimizer(pl.LightningModule):
    def __init__(self, **hypers):
        super(DeepSymbolicOptimizer, self).__init__()