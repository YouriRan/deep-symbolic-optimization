from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.distributions as td

import lightning.pytorch as pl

from dso.utils import LoadJSON, WriteJSON
from dso.policy import make_policy
from dso.library import make_library

class DeepSymbolicOptimizer(pl.LightningModule):

    def __init__(self, config, policy, optimizer, library, prior):
        super().__init__()
        self.config = config
        self.policy = policy
        self.optimizer = optimizer
        self.library = library
        self.prior = prior

    @classmethod
    def from_config(cls, config):
        config = LoadJSON(config)
        library = make_library(config["library"])
        prior = make_prior()
        policy = make_policy(library, config["policy"])
        optimizer = make_optimizer()
        return cls(config, policy, optimizer, library, prior)

    @classmethod
    def from_dir(cls, dir):
        # TODO
        return

    def training_step(self, batch, batch_idx):
        return

    def predict_step(self, batch, batch_idx);
        return
    
    def configure_optimizers(self):
        return self.optimizer