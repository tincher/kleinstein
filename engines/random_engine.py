from .base_engine import BaseEngine
import random


class RandomEngine(BaseEngine):

    def __init__(self):
        pass

    def find_move(self, game):
        return random.choice(game.get_valid_moves())
