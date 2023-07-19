import torch
import torch.nn as nn

class Token(nn.Module):
    def __init__(self, name, arity, complexity, input_var=None):
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

    def forward(self):
        return

    def __repr__(self):
        return self.name

class HardCodedConstant(Token):
    def __init__(self, name, value):
        super().__init__(name, arity=0, complexity=1)
        self.value = value

    def forward(self, x):
        return self.value
