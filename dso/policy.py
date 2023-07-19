import torch.nn as nn

class RNNPolicy(nn.Module):
    def __init__(self, library, cell="lstm", num_layers=1, num_units=32):

        # set up network, with rnn layer and logit layer
        n_tokens = len(library)
        rnn_unit = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_unit(
            input_size=n_tokens,
            hidden_size=num_units,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(n_tokens, n_tokens)
