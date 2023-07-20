import torch
import torch.nn as nn
from dso.library import Token


class Add(Token):

    def __init__(self):
        super().__init__("add", arity=2, complexity=1)

    def forward(self, x, y):
        return torch.add(x, y)


class Subtract(Token):

    def __init__(self):
        super().__init__("sub", arity=2, complexity=1)

    def forward(self, x, y):
        return torch.subtract(x, y)


class Multiply(Token):

    def __init__(self):
        super().__init__("mul", arity=2, complexity=1)

    def forward(self, x, y):
        return torch.multiply(x, y)


class Divide(Token):

    def __init__(self):
        super().__init__("div", arity=2, complexity=2)

    def forward(self, x, y):
        return torch.divide(x, y)


class Sin(Token):

    def __init__(self):
        super().__init__("sin", arity=1, complexity=3)

    def forward(self, x):
        return torch.sin(x)


class Cos(Token):

    def __init__(self):
        super().__init__("cos", arity=1, complexity=3)

    def forward(self, x):
        return torch.cos(x)


class Tan(Token):

    def __init__(self):
        super().__init__("tan", arity=1, complexity=4)

    def forward(self, x):
        return torch.tan(x)


class Exp(Token):

    def __init__(self):
        super().__init__("exp", arity=1, complexity=4)

    def forward(self, x):
        return torch.exp(x)


class Log(Token):

    def __init__(self):
        super().__init__("log", arity=1, complexity=4)

    def forward(self, x):
        return torch.log(x)


class Sqrt(Token):

    def __init__(self):
        super().__init__("sqrt", arity=1, complexity=4)

    def forward(self, x):
        return torch.sqrt(x)


class N2(Token):

    def __init__(self):
        super().__init__("n2", arity=1, complexity=2)

    def forward(self, x):
        return torch.square(x)


class Neg(Token):

    def __init__(self):
        super().__init__("neg", arity=1, complexity=1)

    def forward(self, x):
        return -x


class Abs(Token):

    def __init__(self):
        super().__init__("abs", arity=1, complexity=2)

    def forward(self, x):
        return torch.abs(x)


class Max(Token):

    def __init__(self):
        super().__init__("max", arity=1, complexity=4)

    def forward(self, x):
        return torch.max(x, dim=-1)


class Min(Token):

    def __init__(self):
        super().__init__("min", arity=1, complexity=4)

    def forward(self, x):
        return torch.min(x)


class Tanh(Token):

    def __init__(self):
        super().__init__("tanh", arity=1, complexity=4)

    def forward(self, x):
        return torch.tanh(x)


class Reciprocal(Token):

    def __init__(self):
        super().__init__("inv", arity=1, complexity=2)

    def forward(self, x):
        return torch.reciprocal(x)


class Logabs(Token):

    def __init__(self):
        super().__init__("logabs", arity=1, complexity=4)

    def forward(self, x):
        return torch.log(torch.abs(x))


class Expneg(Token):

    def __init__(self):
        super().__init__("expneg", arity=1, complexity=4)

    def forward(self, x):
        return torch.exp(-x)


class N3(Token):

    def __init__(self):
        super().__init__("n3", arity=1, complexity=3)

    def forward(self, x):
        return torch.pow(x, 3)


class N4(Token):

    def __init__(self):
        super().__init__("n4", arity=1, complexity=3)

    def forward(self, x):
        return torch.pow(x, 4)


class Sigmoid(Token):

    def __init__(self):
        super().__init__("sigmoid", arity=1, complexity=4)

    def forward(self, x):
        return torch.sigmoid(x)


class Langmuir(Token):

    def __init__(self):
        super().__init__("langmuir", arity=1, complexity=4)

    def forward(self, x):
        return torch.divide(x, 1 + x)


unprotected_ops = [
    Add(),
    Subtract(),
    Multiply(),
    Divide(),
    Sin(),
    Cos(),
    Tan(),
    Exp(),
    Log(),
    Sqrt(),
    N2(),
    Neg(),
    Abs(),
    Max(),
    Min(),
    Tanh(),
    Reciprocal(),
    Logabs(),
    Expneg(),
    N3(),
    N4(),
    Sigmoid(),
    Langmuir()
]

function_map = {op.name: op for op in unprotected_ops}
TERMINAL_TOKENS = set([op.name for op in function_map.values() if op.arity == 0])
UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])
