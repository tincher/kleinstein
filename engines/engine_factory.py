from .base_engine import BaseEngine


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
