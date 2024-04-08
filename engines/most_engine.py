import numpy as np

from game_src import Game

from .base_engine import BaseEngine


class MostEngine(BaseEngine):
    """Always take the field which contains the most stones."""

    def __init__(self) -> None:
        pass

    def find_move(self, game: Game) -> int:
        """Find move which uses the most stones in the first pit.

        Parameters
        ----------
        game : Game
            the game for which to find the move

        Returns
        -------
        int
            the index of the pit with the most stones in it
        """
        # get all valid moves
        valid_moves = game.get_valid_moves()

        # whose turn is it
        current_turn = game.top_turn

        # get relevant state
        relevant_state = game.top_state if current_turn else game.bottom_state

        return valid_moves[np.argmax(relevant_state.state[valid_moves])]
