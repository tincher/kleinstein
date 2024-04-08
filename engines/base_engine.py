import abc

from game_src.game import Game


class BaseEngine(abc.ABC):
    """Base Engine to make all engines reusable.

    All subclasses' names should end with 'Engine', this way the EngineFactory can automatically map to them by using
    the first part of the name as a key. E.g. 'count' -> CountEngine
    """

    @abc.abstractmethod
    def find_move(self, game: Game) -> int:
        """Find the move according to the rules of the engine.

        Parameters
        ----------
        game : Game
            game for which to find a move

        Returns
        -------
        int
            predicted move
        """
