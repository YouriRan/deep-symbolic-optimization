import torch
import torch.nn as nn


class Token(nn.Module):
    """
    An arbitrary token or "building block" of a Program object.

    Attributes
    ----------
    name : str
        Name of token.

    arity : int
        Arity (number of arguments) of token.

    complexity : float
        Complexity of token.

    input_var : int or None
        Index of input if this Token is an input variable, otherwise None.

    Methods
    -------
    forward : callable
        Function associated with the token; used for exectuable Programs.
    """

    def __init__(self, name, arity, complexity, input_var=None):
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

        if input_var is not None:
            assert not isinstance(self, Token)

    def forward(self):
        raise NotImplementedError("Base class should not be called")

    def __repr__(self):
        return self.name


class HardCodedConstant(Token):
    """
    A Token with a "value" attribute, whose function returns the value.

    Parameters
    ----------
    value : float
        Value of the constant.
    """

    def __init__(self, name, value):
        super().__init__(name, arity=0, complexity=1)
        self.value = value

    def forward(self, x):
        return self.value


class PlaceolderConstant(Token):
    pass
    """
    A Token for placeholder constants that will be optimized with respect to
    the reward function. The function simply returns the "value" attribute.

    Parameters
    ----------
    value : float
        Value of the constant.
    """

    def __init__(self, name, value):
        super().__init__(name, arity=0, complexity=1)
        self.value = value

    def forward(self, x):
        return self.value


class Polynomial(Token):
    pass


class StateChecker(Token):
    pass


class MultiDiscreteAction(Token):
    pass


def make_library(config):
    return Library.from_config()


class Library():

    def __init__(self, tokens):
        self.tokens = tokens
        self.names = [t.name for t in tokens]
        self.arities = torch.Tensor([t.arity for t in tokens])

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def from_config(cls, config: dict):
        return cls()