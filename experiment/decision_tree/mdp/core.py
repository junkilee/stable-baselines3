"""Contains core classes for the Markov Decision Process (MDP) and State class definitions."""

import numpy as np
# import copy
from abc import ABC, abstractmethod
from experiment.decision_tree.utils.core import sample_from_dict, sample_uniform_from_list


class MDP(ABC):
    """Stateless MDP definition.

    This MDP class does not contain its own state machine.
    """
    def __init__(self, name, actions, gamma=0.99, np_random=None):
        if np_random:
            self._np_random = np_random
        else:
            self._np_random = np.random
        self._name = name
        self._actions = actions
        self._gamma = gamma

    @property
    def name(self):
        return self._name

    @property
    def actions(self):
        return self._actions

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    @abstractmethod
    def start_states(self):
        """Returns either a set of start states or a function that generates a start state among possible candidates."""
        raise NotImplementedError

    @abstractmethod
    def transition_function(self, state, action):
        """Can return the next state, a dictionary of next state and probability pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def reward_function(self, state, action, next_state):
        raise NotImplementedError

    def reset(self):
        init_state = None
        start_states = self.start_states()
        if callable(start_states):
            init_state = start_states()
        elif isinstance(start_states, list):
            init_state = sample_uniform_from_list(start_states, self._np_random)
        elif isinstance(start_states, dict):
            init_state = sample_from_dict(start_states, self._np_random)
        assert isinstance(init_state, State)
        return init_state

    def step(self, state, action):
        next_state = self.transition_function(state, action)
        if isinstance(next_state, dict):
            next_state = sample_from_dict(next_state, self._np_random)
        elif isinstance(next_state, list):
            next_state = sample_uniform_from_list(next_state, self._np_random)
        reward = self.reward_function(state, action, next_state)
        return next_state, reward, next_state.terminal


class State:
    def __init__(self, data, is_terminal=False):
        self._data = data
        self._is_terminal = is_terminal

    @property
    def data(self):
        return self._data

    @property
    def terminal(self):
        return self._is_terminal

    @terminal.setter
    def terminal(self, value):
        self._is_terminal = value

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            return hash(str(self.data))      # numpy arrays
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __str__(self):
        return "S[" + str(self._data) + "]"

    def __eq__(self, other):
        return self.data == other.data
