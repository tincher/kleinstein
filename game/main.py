import numpy as np

TOP = True
BOT = False


class Game(object):

    def __init__(self):
        self.top_state = State(TOP)
        self.bottom_state = State(BOT)
        self.top = True

    def get_valid_moves(self):
        pass

    def get_move_result(self, move):
        state = self.top_state if move.top else self.bottom_state
        move_field = move.field
        if (current_count := state[move.field]) < 2:
            raise ValueError(f"Invalid Move: There are not enough gems in the pit!"
                             f"pit: {move.field}, entry: {state[move.field]}",
                             str(self))

        while current_count > 0:
            # fill pits, check for ring
            # check last, redo
            state.make_move(move_field)

    def get_game_result(self):
        pass

    def __str__(self):
        result = []
        result.extend(self.top_state.get_representation())
        result.extend(self.bottom_state.get_representation())

        result = [entry.replace("[", "").replace("]", "").strip()
                  for entry in result]
        result.insert(2, '-' * len(result[0]))
        result.append("TOP" if self.top else "BOT")
        result = '\n'.join(result)
        return result

    def __repr__(self):
        return str(self)


class State(object):
    def __init__(self, top):
        self.state = np.ones((16), dtype="int") * 2
        self.state[12:] = 0
        self.top = top

    def make_move(self, field):
        """Executes a single move, i.e. take x stones, fill x subsequent pits, returns field + x

        Parameters
        ----------
        field : int
            index of starting pit

        Returns
        -------
        int
            index of last pit filled with a stone
        """
        last_field = (field + (stone_count := self.state[field])) % 16
        self.state[field] = 0
        self.state[field + 1:field + stone_count + 1] += 1
        if last_field < field:
            self.state[:last_field] += 1
        return last_field

    def get_representation(self):
        representation_state = self.state.reshape((2, 8))
        order = [0, 1] if self.top else [1, 0]
        result = []
        result.append(str(representation_state[order[0], ::-1]))
        result.append(str(representation_state[order[1]]))
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
