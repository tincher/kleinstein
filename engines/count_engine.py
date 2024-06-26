import copy

from game_src.game import Game
from game_src.move import Move

from .base_engine import BaseEngine


class CountEngine(BaseEngine):
    """Engine to always use the move that results in the most stones on current side to move."""

    def __init__(self) -> None:
        pass

    def find_move(self, game: Game) -> int:
        """Find the move that results in the most stones on the side which has to move.

        Parameters
        ----------
        game : Game
            game for which to find the move

        Returns
        -------
        int
            the move index which results in the most stones on the current side to move
        """
        # get all valid moves
        valid_moves = game.get_valid_moves()

        # init max_count and index of the maximizing move
        max_stone_count, max_stone_count_index = -1, -1

        # whose turn is it
        current_turn = game.top_turn

        # for all moves
        for index, move in enumerate(valid_moves):
            # take copy of game
            current_game = copy.deepcopy(game)

            # make the move
            current_game.make_move(Move(current_turn, move))

            # get the engine's state
            relevant_state = current_game.top_state if current_turn else current_game.bottom_state

            # how many stones this side has
            stone_count = relevant_state.state.sum()

            # if the current move increases the stone count more than the previously tested moves, store it
            if stone_count > max_stone_count:
                # store stone count
                max_stone_count = stone_count

                # store move index
                max_stone_count_index = index

        # return corresponding move
        return valid_moves[max_stone_count_index]
