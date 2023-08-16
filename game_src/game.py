from .state import State
import numpy as np

TOP = True
BOT = False


class Game(object):

    def __init__(self):
        self.top_state = State(TOP)
        self.bottom_state = State(BOT)
        self.top_turn = True

    def get_valid_moves(self):
        active_state = self.top_state if self.top_turn else self.bottom_state
        return np.where(active_state.state > 1)[0]

    def make_move(self, move):
        if move.top_turn != self.top_turn:
            print("WARNING: move turn is different from game turn")
        active_state, passive_state = self.top_state, self.bottom_state
        if not self.top_turn:
            active_state, passive_state = passive_state, active_state

        move_field = move.field
        initial_move = True
        current_count = self.get_stone_count(move_field, active_state, initial_move)

        while current_count > 1:
            move_field = self.execute_single_move(move_field, active_state, passive_state, initial_move=initial_move)
            current_count = self.get_stone_count(move_field, active_state)
            initial_move = False

        self.top_turn = not self.top_turn

    def execute_single_move(self, move_field, active_state, passive_state, initial_move=False):
        """Executes a single move, i.e. take x stones, fill x subsequent pits, returns field + x

        Parameters
        ----------
        move_field : _type_
            _description_
        active_state : _type_
            _description_
        passive_state : _type_
            _description_

        Returns
        -------
        int
            last field where a stone was put into
        """
        own_stone_count = active_state[move_field]
        enemy_stone_count = 0
        if move_field > 7 and not initial_move:
            opposing_field_index = (15 - (move_field - 8)) % 16
            enemy_stone_count = passive_state[opposing_field_index]
            passive_state[opposing_field_index] = 0
        stone_count = own_stone_count + enemy_stone_count

        last_field = (move_field + stone_count) % 16
        active_state[move_field] = 0
        active_state[move_field + 1:move_field + stone_count + 1] += 1
        if last_field < move_field:
            active_state[:last_field + 1] += 1
        return last_field

    def get_stone_count(self, move, active_state, initial_move=False):
        if (current_count := active_state[move]) < 2 and initial_move:
            raise ValueError(f"Invalid Move: There are not enough stones in the pit!"
                             f"pit: {move}, entry: {active_state[move]}",
                             str(self))

        return current_count

    def get_game_result(self):
        pass

    def __str__(self):
        result = []
        result.extend(self.top_state.get_representation())
        result.extend(self.bottom_state.get_representation())

        result = [entry.replace("[", "").replace("]", "").strip()
                  for entry in result]
        result.insert(2, '-' * 10)
        result.append("TOP" if self.top_turn else "BOT")
        result = '\n'.join(result) + "\n"
        return result

    def __repr__(self):
        return str(self)
