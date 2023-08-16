import numpy as np


class Game(object):

    def __init__(self):
        self.top_state = np.ones((2, 8)) * 2
        self.bottom_state = np.ones((2, 8)) * 2
        self.top_state[1, 4:] = 0
        self.bottom_state[1, 4:] = 0
        self.top = True

    def get_valid_moves(self):
        pass

    def get_move_result(self, move):
        pass

    def get_game_result(self):
        pass

    def __str__(self):
        result = []
        result.append(str(self.top_state[0, ::-1]))
        result.append(str(self.top_state[1]))
        result.append(str(self.bottom_state[1, ::-1]))
        result.append(str(self.bottom_state[0]))

        result = [entry.replace("[", "").replace("]", "").strip() for entry in result]
        result.insert(2, '-' * len(result[0]))
        result.append("TOP" if self.top else "BOT")
        result = '\n'.join(result)
        return result

    def __repr__(self):
        return str(self)


class Move(object):
    def __init__(self, top, field):
        """Init a new move

        Parameters
        ----------
        top : bool
            whether the move is for the top or bottom player
        field : int
            which field the move starts with (counting from bottom-left from player-view in play-direction)
        """
        pass


def generate_valid_moves(self, game, player):
    pass
