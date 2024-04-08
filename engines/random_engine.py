import random

from game_src import Game

from .base_engine import BaseEngine


class RandomEngine(BaseEngine):
    """Engine which makes random moves."""

    def __init__(self) -> None:
        pass

    def find_move(self, game: Game) -> int:
        """Find a random move.

        Parameters
        ----------
        game : Game
            game for which to find the move

        Returns
        -------
        int
            a random move index
        """
        return random.choice(game.get_valid_moves())
