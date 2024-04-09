from .base_engine import BaseEngine
from .count_engine import CountEngine  # noqa: F401 these need to be imported so the __subclasses__ call can work
from .most_engine import MostEngine  # noqa: F401
from .random_engine import RandomEngine  # noqa: F401
from .steal_engine import StealEngine  # noqa: F401
from .td_engine import TDEngine  # noqa: F401


class EngineFactory:
    """Factory class to choose the correct engine (subclass from BaseEngine) by its given name.

    The name should be the first part of its class name, after removing the string 'engine' from its class name.
    """

    def __init__(self) -> None:
        """Initialize the mapping from names to class."""
        self._mapping = {}
        for subclass in BaseEngine.__subclasses__():
            self._mapping[subclass.__name__.lower().replace("engine", "")] = subclass

    def get_engine(self, name: str) -> BaseEngine:
        """Provide the class fitting to the given name.

        Parameters
        ----------
        name : str
            name of the engine

        Returns
        -------
        BaseEngine
            the fitting subclass of BaseEngine
        """
        return self._mapping[name]
