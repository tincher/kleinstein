import numpy as np
from .base_engine import BaseEngine
from td_stones.model import TDStones
from game_src import Move
import torch
import copy


class TDEngine(BaseEngine):

    def __init__(self, model_path=None):
        self.model = TDStones(40, 32)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def find_move(self, game):
        best_move, _, _ = predict_best_move(game, self.model)

        return best_move


def predict_best_move(game, model):
    valid_moves = game.get_valid_moves()

    best_output, best_output_index, best_game_representation = -1, -1, None

    current_turn = game.top_turn

    for move_index, move in enumerate(valid_moves):

        current_game = copy.deepcopy(game)

        current_game.make_move(Move(current_turn, move))

        own_state = current_game.top_state if current_turn else current_game.bottom_state
        enemy_state = current_game.top_state if not current_turn else current_game.bottom_state

        game_representation = torch.tensor(np.stack((own_state.state, enemy_state.state)), dtype=torch.float).reshape((-1))
        model_output = model(game_representation)

        if (own_win_probability := model_output[0]) > best_output:
            best_output = own_win_probability
            best_output_index = move_index
            best_game_representation = game_representation

    return valid_moves[best_output_index], best_game_representation, best_output
