import copy

import numpy as np
import torch

from game_src.game import Game
from game_src.move import Move
from td_stones.model import TDStones

from .base_engine import BaseEngine


class TDEngine(BaseEngine):
    """Use a model to predict the next move."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the model, load state dict from given path.

        Parameters
        ----------
        model_path : str | None, optional
            path to a model filee, by default None
        """
        self.model = TDStones(40, 32)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def find_move(self, game: Game) -> int:
        """Find the move that results in the best prediction for the current side to move.

        Parameters
        ----------
        game : Game
            game for which to find the move

        Returns
        -------
        int
            the move index which results in the best prediction for the current move
        """
        best_move, _ = self.predict_best_move(game, self.model)

        return best_move

    @classmethod
    def predict_best_move(cls, game: Game, model: TDStones) -> tuple[int, torch.Tensor]:
        """Predict the best move for the side to move.

        The best move is the move which results in the highest (top) or lowest (bot) predicted score.

        Parameters
        ----------
        game : Game
            game for which to predict the best move

        Returns
        -------
        tuple[int, torch.Tensor, torch.Tensor]
            best move, representation of the best game, best score
        """
        valid_moves = game.get_valid_moves()

        best_output, best_output_index = -1, -1

        current_turn = game.top_turn

        for move_index, move in enumerate(valid_moves):
            current_game = copy.deepcopy(game)

            current_game.make_move(Move(current_turn, move))

            game_representation = torch.tensor(
                np.stack((current_game.top_state.state, current_game.bottom_state.state)),
                dtype=torch.float,
            ).reshape(-1)
            model_output = model(game_representation)

            if (own_win_probability := model_output[0]) > best_output:
                best_output = own_win_probability
                best_output_index = move_index

        return valid_moves[best_output_index], best_output
