from enum import Enum

import numpy as np

MAX_STONE_COUNT = 2


class Turn(Enum):
    TOP = True
    BOTTOM = False


class State:
    def __init__(self, top: Turn) -> None:
        self.state = np.ones((16), dtype="int") * 2
        self.state[12:] = 0
        self.top = top

    def get_representation(self) -> list[str]:
        representation_state = self.state.reshape((2, 8))
        order = [0, 1] if self.top else [1, 0]
        result = []
        result.append(str(representation_state[order[0], ::-1]))
        result.append(str(representation_state[order[1]]))
        return result

    def is_lost(self) -> None:
        front_row_empty = self.state[8:].sum() < 1
        no_legal_moves = self.state.max() < MAX_STONE_COUNT
        return front_row_empty or no_legal_moves

    def __getitem__(self, index: int) -> int:
        return self.state[index]

    def __setitem__(self, index: int, value: int) -> None:
        self.state[index] = value
