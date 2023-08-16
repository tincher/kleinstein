from .state import State

TOP = True
BOT = False


class Game(object):

    def __init__(self):
        self.top_state = State(TOP)
        self.bottom_state = State(BOT)
        self.top_turn = True

    def get_valid_moves(self):
        pass

    def make_moves(self, move):
        if move.top_turn != self.top_turn:
            print("WARNING: move turn is not same as game turn")
        state = self.top_state if self.top_turn else self.bottom_state
        move_field = move.field
        if (current_count := state.state[move.field]) < 2:
            raise ValueError(f"Invalid Move: There are not enough stones in the pit!"
                             f"pit: {move.field}, entry: {state[move.field]}",
                             str(self))

        while current_count > 1:
            move_field = state.make_move(move_field)
            current_count = state.state[move_field]

        self.top_turn = not self.top_turn

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
