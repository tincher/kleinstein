from game.main import Game


def test_init_print():
    game = Game()
    expected_output = '2. 2. 2. 2. 2. 2. 2. 2.\n2. 2. 2. 2. 0. 0. 0. 0.\n-----------------------\n0. 0. 0. 0. 2. 2. 2. 2.\n2. 2. 2. 2. 2. 2. 2. 2.\nTOP'
    str(game) == expected_output


def test_move():
    pass
