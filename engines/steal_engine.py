import copy

from game_src.game import Game
from game_src.move import Move

from .base_engine import BaseEngine


class StealEngine(BaseEngine):
    """Engine with the goal of the enemy having the least stones."""

    def __init__(self) -> None:
        """Initialize the StealEngine."""

    def find_move(self, game: Game) -> int:
        """Find the move that results in the least stones on the non-moving side.

        Parameters
        ----------
        game : Game
            game for which to find the move

        Returns
        -------
        int
            the move index which results in the least stones on the current side not to move
        """
        # get all valid moves
        valid_moves = game.get_valid_moves()

        # init max_count and index of the maximizing move
        min_stone_count, min_stone_count_index = 100, -1

        # whose turn is it
        current_turn = game.top_turn

        # for all moves
        for index, move in enumerate(valid_moves):
            # take copy of game
            current_game = copy.deepcopy(game)

            # make the move
            current_game.make_move(Move(current_turn, move))

            # get the enemy's state
            relevant_state = current_game.top_state if not current_turn else current_game.bottom_state

            # how many stones this side has
            stone_count = relevant_state.state.sum()

            # if the current move decreses the enemy's stone count more than the previously tested moves, store it
            if stone_count < min_stone_count:
                # store stone count
                min_stone_count = stone_count

                # store move index
                min_stone_count_index = index

        # return corresponding move
        return valid_moves[min_stone_count_index]
