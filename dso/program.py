import torch
import torch.nn as nn


class Program(nn.Module):

    def __init__(self, tokens):
        self.tokens = tokens

    def forward(self, x):
        for token in self.tokens:
            x = token(x)
        return x