import torch
import torch.nn as nn
from dso.library import Token

class Add(Token):
    def __init__(self):
        super().__init__("add", arity=2, complexity=1)
    
    def forward(self, x):
        return torch.sum(x, dim=-1)
    
class Subtract(Token):
    def __init__(self):
        super().__init__("sub", arity=2, complexity=1)
    
    def forward(self, x):
        return -torch.diff(x, dim=-1)

class Multiply(Token):
    def __init__(self):
        super().__init__("mul", arity=2, complexity=1):
        
    def forward(self, x):
        return torch.prod(x, dim=-1)

class Divide(Token):
    def __init__(self):
        super().__init__("div", arity=2, complexity=2)

    def forward(self, x):
        return torch.divide(x[..., 0], x[..., 1])

class Sin(Token):
    def __init__(self):
        super().__init__("sin", arity=1, complexity=3)

    def forward(self, x):
        return torch.sin(x)