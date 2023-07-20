import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from dso.library import Library


def make_policy(library: Library, config: dict):
    return RNNPolicy.from_config(config)


class RNNPolicy(nn.Module):

    def __init__(self, library: Library, action_prob_lowerbound, cell="lstm", num_layers=1, num_units=32):
        self.library = library
        self.action_prob_lowerbound = action_prob_lowerbound

        # set up network, with rnn layer and logit layer
        rnn_unit = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        rnn = rnn_unit(input_size=self.n_tokens, hidden_size=num_units, num_layers=num_layers, batch_first=True)
        linear = nn.Linear(len(self.library), len(self.library))
        self.cell = nn.Sequential(rnn, linear)

    @classmethod
    def from_config(cls, library: Library, config: dict):
        return cls(library)

    def forward(self, x):
        return

    def make_neglogp_and_entropy(self, B, entropy_gamma):
        return

    def sample(self, n):
        return

    def sample_novel(self, n):
        return

    def compute_probs(self, memory_batch, log=False):
        return

    def apply_action_prob_lowerbound(self, logits):
        probs = F.softmax(logits, dim=-1)
        probs_bounded = ((1 - self.action_prob_lowerbound) * probs +
                         self.action_prob_lowerbound / float(len(self.library)))
        logits_bounded = torch.log(probs_bounded)
        return logits_bounded
