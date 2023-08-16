from .random_engine import RandomEngine


class EngineFactory():
    def __init__(self):
        pass

    def get_engine(self, name):
        return RandomEngine
