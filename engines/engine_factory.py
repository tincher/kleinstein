from .random_engine import RandomEngine
from .count_engine import CountEngine
from .base_engine import BaseEngine


class EngineFactory():
    def __init__(self):
        self._mapping = {}
        for subclass in BaseEngine.__subclasses__():
            self._mapping[subclass.__name__.lower().replace("engine", "")] = subclass

    def get_engine(self, name):
        return self._mapping[name]
