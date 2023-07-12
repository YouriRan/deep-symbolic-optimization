from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from dso.program import Program


class StateManager(ABC):
    """
    An interface for handling the torch.Tensor inputs to the Policy.
    """

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform the needed initializations (e.g., if the torch context is needed)
        :param policy the policy class
        """
        self.policy = policy
        self.max_length = policy.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Policy, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : torch.Tensor (dtype=torch.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : torch.Tensor (dtype=torch.float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {"hierarchical": HierarchicalStateManager}

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self,
                 observe_parent=True,
                 observe_sibling=True,
                 observe_action=False,
                 observe_dangling=False,
                 embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setup_manager(self, policy):
        super().setup_manager(policy)
        # Create embeddings if needed
        # Create embeddings if needed
        if self.embedding:
            self.action_embeddings = nn.Embedding(self.library.n_action_inputs, self.embedding_size).to(self.device)
            self.parent_embeddings = nn.Embedding(self.library.n_parent_inputs, self.embedding_size).to(self.device)
            self.sibling_embeddings = nn.Embedding(self.library.n_sibling_inputs, self.embedding_size).to(self.device)

    def get_tensor_input(self, obs):
        observations = []
        action, parent, sibling, dangling = obs.split(1, dim=1)

        # Cast action, parent, sibling to int for embedding_lookup or one_hot
        action = action.long()
        parent = parent.long()
        sibling = sibling.long()

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings(action)
            else:
                x = F.one_hot(action, num_classes=self.library.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings(parent)
            else:
                x = F.one_hot(parent, num_classes=self.librar.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings(sibling)
            else:
                x = F.one_hot(sibling, num_classes=self.library.n_sibling_inputs)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = dangling.unsqueeze(-1)
            observations.append(x)

        input_ = torch.cat(observations, -1)
        return input_
