"""Value Iteration Class"""
from causal_rl.planners.core import Planner
from collections import defaultdict
from causal_rl.utils import EPSILON4
from causal_rl.envs.mdp import State
import queue


class ValueIteration(Planner):
    """

    """

    def __init__(self, mdp, max_iteration=10000, delta=EPSILON4):
        self._max_iteration = max_iteration
        self._transition_matrix = None
        self._reachable_states = None
        self._value_table = None
        self._mdp = mdp
        self._delta = delta
        super().__init__("ValueIteration")

    @property
    def is_states_all_covered(self):
        return self._reachable_states is None

    @property
    def is_transition_matrix_made(self):
        return self._transition_matrix is None

    @property
    def is_ran(self):
        return self._value_table is not None

    @property
    def transition_matrix(self):
        if self._transition_matrix is None:
            self._build_transition_matrix()
        return self._transition_matrix

    @property
    def reachable_states(self):
        if self._reachable_states is None:
            self._traverse_all_reachable_states(self._mdp.start_states())
        return self._reachable_states

    def _traverse_all_reachable_states(self, start_states):
        assert not callable(start_states)  # do not cover the case where start states need to be sampled
        if self._reachable_states is None:
            self._reachable_states = set([])

            for start_state in start_states:
                if start_state not in self._reachable_states:
                    state_queue = queue.Queue()
                    state_queue.put(start_state)

                    while not state_queue.empty():
                        cur_state = state_queue.get()
                        for action in self._mdp.actions:
                            next_states = self._mdp.transition_function(cur_state, action)
                            if isinstance(next_states, State):
                                if next_states not in self._reachable_states:
                                    self._reachable_states.add(next_states)
                                    state_queue.put(next_state)
                            else:
                                for next_state in next_states:
                                    add = False
                                    if isinstance(next_states, dict):
                                        if next_states[next_state] > 0.0:
                                            add = True
                                    else:
                                        add = True
                                    if add and next_state not in self._reachable_states:
                                        self._reachable_states.add(next_state)
                                        state_queue.put(next_state)

    def _build_transition_matrix(self):
        if self._transition_matrix is None:
            self._transition_matrix = defaultdict(lambda: defaultdict(dict))
            for state in self._reachable_states:
                for action in self._mdp.actions:
                    next_states = self._mdp.transition_function(state, action)
                    if isinstance(next_states, dict):
                        self._transition_matrix[state][action] = next_states
                    elif isinstance(next_states, State):
                        self._transition_matrix[state][action] = {next_states: 1.0}
                    else:
                        raise TypeError

    def run(self):
        max_diff = float("inf")
        num_iterations = 0

        self._traverse_all_reachable_states(self._mdp.start_states())
        self._build_transition_matrix()
        self._value_table = defaultdict(float)

        while max_diff > self._delta and num_iterations < self._max_iteration:
            max_diff = 0.0
            for state in self._reachable_states:
                if state.terminal:
                    continue
                max_q = float("-inf")
                for action in self._mdp.actions:
                    max_q = max(max_q, self.get_q_value(state, action))
                max_diff = max(max_diff, abs(self._value_table[state] - max_q))
                self._value_table[state] = max_q
            num_iterations += 1

    def get_value(self, state):
        if not isinstance(state, State):
            raise TypeError
        return self._value_table[state]

    def get_q_value(self, state, action):
        q_s_a = 0.0
        for next_state in self._transition_matrix[state][action].keys():
            q_s_a += self._transition_matrix[state][action][next_state] * \
                     (self._mdp.reward_function(state, action, next_state) +
                      self._mdp.gamma * self._value_table[next_state])
        return q_s_a

    def policy(self, state):
        best_action, max_q = self._get_max_q_action(state)
        return best_action

    def _get_max_q_action(self, state):
        best_action = None
        max_q = float("-inf")
        for action in self._mdp.actions:
            q_s_a = self.get_q_value(state, action)
            if max_q < q_s_a:
                best_action = action
                max_q = q_s_a
        return best_action, max_q

    def plan(self, state=None, horizon=100):
        if state is None:
            state = self._mdp.reset()
        num_steps = 0
        action_sequence = []
        state_sequence = [state]
        rew_sequence = []
        while not state.terminal and num_steps < horizon:
            action = self.policy(state)
            next_state, rew, terminal = self._mdp.step(state, action)
            action_sequence.append(action)
            state_sequence.append(next_state)
            rew_sequence.append(rew)
            state = next_state
            num_steps += 1
        return state_sequence, action_sequence, rew_sequence
