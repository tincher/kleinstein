from .base_engine import BaseEngine
import copy
from game_src import Move


class StealEngine(BaseEngine):

    def __init__(self):
        pass

    def find_move(self, game):
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
