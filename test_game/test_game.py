from game_src import Game, Move
from game_src.state import State


def test_init_print():
    game = Game()
    expected_output = '2 2 2 2 2 2 2 2\n2 2 2 2 0 0 0 0\n----------\n0 0 0 0 2 2 2 2\n2 2 2 2 2 2 2 2\nTOP\n'
    assert str(game) == expected_output


def test_make_moves():
    game = Game()
    move = Move(True, 0)
    game.make_move(move)

    expected_output = '3 3 0 3 3 0 3 0\n0 3 3 0 1 1 1 0\n----------\n0 0 0 0 2 2 2 2\n2 2 2 2 2 2 2 2\nBOT\n'
    assert str(game) == expected_output


def test_make_moves_spillover():
    game = Game()
    move = Move(True, 0)
    game.make_move(move)
    move = Move(False, 0)
    game.make_move(move)
    move = Move(True, 10)
    game.make_move(move)

    expected_output = '3 3 0 3 3 1 4 1\n0 3 0 1 1 0 2 1\n----------\n1 1 1 1 0 0 3 0\n0 3 0 3 3 0 3 3\nBOT\n'
    assert str(game) == expected_output
