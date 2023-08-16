import numpy as np


class Game(object):

    def __init__(self):
        self.top_state = State()
        self.bottom_state = State()
        self.top = True

    def get_valid_moves(self):
        pass

    def get_move_result(self, move):
        state = self.top_state if move.top else self.bottom_state
        current_count = state[move.field]
        if current_count < 2:
            raise ValueError(f"There are not enough gems in the pit, pit: {move.field}, entry: {state[move.field]}",
                             str(self))

        while current_count > 0:
            # fill pits, check for ring
            # check last, redo
            pass

    def get_game_result(self):
        pass

    def __str__(self):
        result = []
        result.extend(self.top_state.get_representation(True))
        result.extend(self.bottom_state.get_representation(False))

        result = [entry.replace("[", "").replace("]", "").strip() for entry in result]
        result.insert(2, '-' * len(result[0]))
        result.append("TOP" if self.top else "BOT")
        result = '\n'.join(result)
        return result

    def __repr__(self):
        return str(self)


class State(object):
    def __init__(self):
        self.state = np.ones((2, 8)) * 2
        self.state[1, 4:] = 0

    def make_move(self, field):
        pass

    def get_representation(self, top=False):
        order = [0, 1] if top else [1, 0]
        result = []
        result.append(str(self.state[order[0], ::-1]))
        result.append(str(self.state[order[1]]))
        return result


class Move(object):

    __slots__ = ["top", "field"]

    def __init__(self, top, field):
        """Init a new move

        Parameters
        ----------
        top : bool
            whether the move is for the top or bottom player
        field : int
            which field the move starts with (counting from bottom-left from player-view in play-direction)
        """
        self.top = top
        self.field = field


def generate_valid_moves(self, game, player):
    pass
