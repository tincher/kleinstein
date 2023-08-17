import numpy as np
from .base_engine import BaseEngine
from game_src import Move


class MostEngine(BaseEngine):

    def __init__(self):
        pass

    def find_move(self, game):
        # get all valid moves
        valid_moves = game.get_valid_moves()

        # whose turn is it
        current_turn = game.top_turn

        # get relevant state
        relevant_state = game.top_state if current_turn else game.bottom_state

        return valid_moves[np.argmax(relevant_state.state[valid_moves])]
