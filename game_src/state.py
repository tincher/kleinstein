import numpy as np


class State(object):
    def __init__(self, top):
        self.state = np.ones((16), dtype="int") * 2
        self.state[12:] = 0
        self.top = top

    def get_representation(self):
        representation_state = self.state.reshape((2, 8))
        order = [0, 1] if self.top else [1, 0]
        result = []
        result.append(str(representation_state[order[0], ::-1]))
        result.append(str(representation_state[order[1]]))
        return result

    def is_lost(self):
        front_row_empty = self.state[8:].sum() < 1
        no_legal_moves = self.state.max() < 2
        return front_row_empty or no_legal_moves

    def __getitem__(self, index):
        return self.state[index]

    def __setitem__(self, index, value):
        self.state[index] = value
