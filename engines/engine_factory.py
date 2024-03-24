from .base_engine import BaseEngine
from .count_engine import CountEngine
from .most_engine import MostEngine
from .random_engine import RandomEngine
from .steal_engine import StealEngine
from .td_engine import TDEngine


class EngineFactory():
    def __init__(self):
        self._mapping = {}
        for subclass in BaseEngine.__subclasses__():
            self._mapping[subclass.__name__.lower().replace("engine", "")] = subclass

    def get_engine(self, name):
        return self._mapping[name]
