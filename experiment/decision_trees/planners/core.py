"""Contains core classes for planners

Code structures are borrowed from simple_rl(https://github.com/david-abel/simple_rl)
"""
from abc import ABC, abstractmethod


class Planner(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def plan(self, state, horizon):
        pass

    @abstractmethod
    def policy(self, state):
        pass

    def __str__(self):
        return self._name
