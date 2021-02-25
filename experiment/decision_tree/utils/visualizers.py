"""Abstract Classes for visualizers, graph plotters."""
import tkinter as tk
from abc import ABC, abstractmethod


class Visualizer(ABC):
    @abstractmethod
    def run(self):
        """Runs the main UI loop."""
        raise NotImplementedError


class GraphPlotter(ABC):
    def __init__(self, in_memory=True):
        self._in_memory = in_memory
        pass

    @property
    def in_memory(self):
        return self._in_memory

    @abstractmethod
    def draw(self):
        raise NotImplementedError

    @abstractmethod
    def show(self, block=True):
        """Initializes and shows the window and runs the main UI loop until there is a key input."""
        raise NotImplementedError
