"""Finite MDP Class for the Cart Pole domain in Gym.
"""

from experiment.gym_wrappers.cart_pole import ModifiedCartPoleEnv
from experiment.decision_trees.mdp import MDP, State
from collections import defaultdict
import numpy as np


class Range(object):    
    def __init__(self, name, range_list):
        self._name = name        
        self._range = []
        self._range = np.array([])
        first = True
        for _min, _max, _num in range_list:
            linsp = np.linspace(_min, _max, _num)
            if not first:
                linsp = linsp[1:]
            self._range = np.concatenate((self._range, linsp))
            first = False                

    @property
    def name(self):
        return self._name
    
    @property
    def range_size(self):
        return len(self._range)

    @property
    def range(self):
        return self._range

    def retrieve_features(self, obs):
        frs = []
        for j, x in enumerate(self._range):
            if obs <= x:
                frs += [True]
            else:
                frs += [False]
        return frs[1:] # discard the first one
    
    def get_index(self, obs):
        idx = -1
        for j, x in enumerate(self._range):
            if obs <= x:
                idx = j
                break
        return idx if idx != -1 else self.range_size - 1
    
    def get_value(self, idx):
        return (self._range[idx - 1] + self._range[idx]) / 2    # Simple Interpolation


class CartPoleRangeSet(object):
    VAR_NAMES = ['lat_pos', 'lat_vel', 'ang_pos', 'ang_vel']
    ALT_NAMES = ['x', 'x_dot', 'theta', 'theta_dot']
    LATEX_NAMES = ['x', '\\dot{x}', '\\theta', '\\dot{\\theta}']

    def __init__(self, ranges):
        self._ranges = []
        for i, range in ranges:
            self._ranges.append(Range(CartPoleRangeSet.VAR_NAMES[i], range))

    @property
    def ranges(self):
        return self._ranges

    def convert_to_features(self, obs):
        feature_outputs = []
        for i, _range in enumerate(self._ranges):
            features_outputs += [_range.retrieve_features(obs[i])]
        return feature_outputs

    def convert_to_indices(self, obs):
        indices = []
        for i, _range in enumerate(self._ranges):
            indices += [_range.get_index(obs[i])]
    
    def convert_to_values(self, indices):
        values = []
        for i, _range in enumerate(self._ranges):
            values += [_range.get_value(indices[i])]
        return values


default_range_set = CartPoleRangeSet([
        [[-2.4, 2.4, 21]],      # lat. pos.
        [[-4.8, 4.8, 21]],      # lat. vel.
        [[-0.209, 0.209, 21]],  # pole ang.
        [[-0.418, 0.418, 21]]   # pole ang. vel.
    ])


class FiniteCartPoleState(State):
    def __init__(self, values, is_terminal=False, range_set=None):        
        self._range_set = default_range_set if range_set is None else range_set
        data = self._range_set.convert_to_indices(values)
        super().__init__(data, is_terminal)

    @property
    def values(self):
        return self._range_set.convert_to_values(self.data)

    @property
    def indices(self):
        return self.data    

    def __str__(self):
        return "S[" + str(self.data) + "-" + str(self.values) + "]"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, FiniteCartPoleState) and self.data == other.data

    def __hash__(self):
        return super().__hash__()


class FiniteCartPoleMDP(MDP):
    ACTIONS = ["left", "right"]    

    def __init__(self, start_states=None, step_cost=0.0,
                 gamma=0.99, seed=None, range_set=None):
        
        self._orig_env = ModifiedCartPoleEnv ()
        self._start_states = start_states
        self._range_set = None        
        if seed is not None:
            np_random = np.random.default_rng()
            np_random.seed(seed)
        else:
            np_random = None
        super().__init__("discretized_cartpole", FiniteCartPoleMDP.ACTIONS, gamma, np_random=np_random)


    def sample_start_states(self):
        state = self.create_state(self.orig_env.sample_start_state())
        return state

    def create_state(self, values):
        state = FiniteCartPoleState(values, range_set=self._range_set)
        if self.orig_env.check_terminal_state(state):
            state.terminal = True
        return state

    def start_states(self):
        if not self._start_states:
            return self.sample_start_states
        if not self._start_states:
            if isinstance(self._start_locations, list):
                self._start_states = []
                for location in self._start_locations:
                    self._start_states.append(self.create_state(location))
            elif isinstance(self._start_locations, dict):
                self._start_states = {}
                for location in self._start_locations:
                    self._start_states[self.create_state(location)] = self._start_locations[location]
        return self._start_states

    def reward_function(self, state, action, next_state=None):
        return 1.0

    def transition_function(self, state, action):
        """

        Returns:
            Dict(str, float): a dictionary mapping states to probabilities
        """        
        result_dict = defaultdict(float)
        next_state = self._orig_env.get_next_state(state, FiniteCartPoleMDP.ACTIONS.index(action))
        result_dict[next_state] = 1.0
        return result_dict


if __name__ == '__main__':
    mdp = FiniteCartPoleMDP()
    from experiment.decision_trees.planners.value_iteration import ValueIteration
    vi = ValueIteration(mdp)
    start = FiniteCartPoleState((0, 0))
    vi.run()
    states, actions, rews = vi.plan(start, horizon=10)