import numpy as np


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
            self.state[:last_field+1] += 1
        return last_field

    def get_representation(self):
        representation_state = self.state.reshape((2, 8))
        order = [0, 1] if self.top else [1, 0]
        result = []
        result.append(str(representation_state[order[0], ::-1]))
        result.append(str(representation_state[order[1]]))
        return result
