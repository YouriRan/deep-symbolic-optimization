"""Controller used to generate distribution over hierarchical, variable-length objects."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch

from dso.policy import Policy
from dso.utils import make_batch_ph



def safe_cross_entropy(p, logq, dim=-1):
    """Compute p * logq safely, by susbstituting
    logq[index] = 1 for index such that p[index] == 0
    """
    # Put 1 where p == 0. In the case, q =p, logq = -inf and this
    # might procude numerical errors below
    safe_logq = torch.where(p > 0.0, torch.ones_like(logq), logq)
    # Safely compute the product
    return -torch.sum(p * safe_logq, dim=dim)


class RNNPolicy(Policy):
    """Recurrent neural network (RNN) policy used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees.

    Parameters
    ----------
    action_prob_lowerbound: float
        Lower bound on probability of each action.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    max_attempts_at_novel_batch: int
        maximum number of repetitions of sampling to get b new samples
        during a call of policy.sample(b)

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer. 

    sample_novel_batch: bool
        if True, then a call to policy.sample(b) attempts to produce b samples
        that are not contained in the cache

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.
        
    """

    def __init__(
            self,
            prior,
            state_manager,
            debug=0,
            max_length=30,
            action_prob_lowerbound=0.0,
            max_attempts_at_novel_batch=10,
            sample_novel_batch=False,
            # RNN cell hyperparameters
            cell='lstm',
            num_layers=1,
            num_units=32,
            initializer='zeros'):
        super().__init__(prior, state_manager, debug, max_length)
        self.action_prob_lowerbound = action_prob_lowerbound

        # len(tokens) in library
        self.n_choices = Program.library.L

        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch
        self.batch_size = torch.empty()

        # omitted zero initialization (no symmetry breaking)
        self.cell = cell
        if self.cell == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.state_manager.state_dim,
                hidden_size=num_units,
                num_layers=num_layers,
                batch_first=True
            )
        elif self.cell == "gru":
            self.rnn = nn.GRU(
                input_size=self.state_manager.state_dim,
                hidden_size=num_units,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError("cell needs to be lstm or gru")
        self.fc = nn.Linear(in_features=num_units, out_features=self.prior.n_choices)

    def set_up_model(self):
        return

    def forward(self, obs, lengths):
        packed_obs = nn.utils.rnn.pack_padded_sequence(obs, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.rnn(packed_obs)
        outputs, _ = nn.utils.rnn.pack_padded_sequence(packed_outputs, batch_first=True)
        logits = self.fc(outputs)
        return logits
    
    def loop_fn(self, time, cell_output, cell_state, loop_state):
        if cell_output is None:
            finished = torch.zeros(size=[self.batch_size], )
        return finished, next_input, next_cell_state, emit_output, next_loop_state

    def make_neglogp_and_entropy(self, actions, probs, lengths):
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.

        Returns
        -------
        neglogp, entropy :
            Tensorflow tensors
        """
        m = Categorical(probs)
        neglogp = -m.log_prob(actions)
        entropy = m.entropy()
        return neglogp, entropy

    def sample(self, n: int):
        """Sample batch of n expressions

        Returns
        -------
        actions, obs, priors : 
            Or a batch
        """
        if self.sample_novel_batch:
            actions, obs, priors = self.sample_novel(n)
        else:
            feed_dict = {self.batch_size: n}
            actions, obs, priors = self.sess.run([self.actions, self.obs, self.priors], feed_dict=feed_dict)

        return actions, obs, priors

    def sample_novel(self, n: int):
        """Sample a batch of n expressions not contained in cache.

        If unable to do so within self.max_attempts_at_novel_batch,
        then fills in the remaining slots with previously-seen samples.

        Parameters
        ----------
        n: int
            batch size

        Returns
        -------
        unique_a, unique_o, unique_p: np.ndarrays
        """
        feed_dict = {self.batch_size: n}
        n_novel = 0
        # Keep the samples that are produced by policy and already exist in cache,
        # so that DSO can train on everything
        old_a, old_o, old_p = [], [], []
        # Store the new samples separately for (expensive) reward evaluation
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            # [batch, time], [batch, obs_dim, time], [batch, time, n_choices]
            actions, obs, priors = self.sess.run([self.actions, self.obs, self.priors], feed_dict=feed_dict)
            n_attempts += 1
            new_indices = []  # indices of new and unique samples
            old_indices = []  # indices of samples already in cache
            for idx, a in enumerate(actions):
                # tokens = Program._finish_tokens(a)
                tokens = _finish_tokens(a)
                key = tokens.tostring()
                if not key in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)
            # get all new actions, obs, priors in this group
            new_a.append(np.take(actions, new_indices, axis=0))
            new_o.append(np.take(obs, new_indices, axis=0))
            new_p.append(np.take(priors, new_indices, axis=0))
            old_a.append(np.take(actions, old_indices, axis=0))
            old_o.append(np.take(obs, old_indices, axis=0))
            old_p.append(np.take(priors, old_indices, axis=0))

        # number of slots in batch to be filled in by redundant samples
        n_remaining = n - n_novel

        # -------------------- combine all -------------------- #
        # Pad everything to max_length
        for tup, name in zip([(old_a, new_a), (old_o, new_o), (old_p, new_p)], ['action', 'obs', 'prior']):
            dim_length = 1 if name in ['action', 'prior'] else 2
            max_length = np.max([list_batch.shape[dim_length] for list_batch in tup[0] + tup[1]])
            # tup is a tuple of (old_?, new_?), each is a list of batches
            for list_batch in tup:
                for idx, batch in enumerate(list_batch):
                    n_pad = max_length - batch.shape[dim_length]
                    # Pad with 0 for everything because training step
                    # truncates based on each sample's own sequence length
                    # so the value does not matter
                    if name == 'action':
                        width = ((0, 0), (0, n_pad))
                        vals = ((0, 0), (0, 0))
                    elif name == 'obs':
                        width = ((0, 0), (0, 0), (0, n_pad))
                        vals = ((0, 0), (0, 0), (0, 0))
                    else:
                        width = ((0, 0), (0, n_pad), (0, 0))
                        vals = ((0, 0), (0, 0), (0, 0))
                    list_batch[idx] = np.pad(batch, pad_width=width, mode='constant', constant_values=vals)

        old_a = np.concatenate(old_a)
        old_o = np.concatenate(old_o)
        old_p = np.concatenate(old_p)
        # If not enough novel samples, then fill in with redundancies
        new_a = np.concatenate(new_a + [old_a[:n_remaining]])
        new_o = np.concatenate(new_o + [old_o[:n_remaining]])
        new_p = np.concatenate(new_p + [old_p[:n_remaining]])

        # first entry serves to force object type, and also
        # indicates not to use it if zero
        self.extended_batch = np.array([old_a.shape[0], old_a, old_o, old_p], dtype=object)
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""

        feed_dict = {self.memory_batch_ph: memory_batch}

        if log:
            fetch = self.memory_logps
        else:
            fetch = self.memory_probs
        probs = self.sess.run([fetch], feed_dict=feed_dict)[0]
        return probs

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to probabilities of each action.

        Parameters
        ----------
        logits: tf.Tensor where last dimension has size self.n_choices

        Returns
        -------
        logits_bounded: tf.Tensor
        """
        probs = tf.nn.softmax(logits, axis=-1)
        probs_bounded = ((1 - self.action_prob_lowerbound) * probs +
                         self.action_prob_lowerbound / float(self.n_choices))
        logits_bounded = tf.log(probs_bounded)

        return logits_bounded
