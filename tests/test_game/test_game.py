import numpy as np

from game_src import Game, Move


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


def test_make_moves_spillover_includes_stealing():
    game = Game()
    move = Move(True, 0)
    game.make_move(move)
    move = Move(False, 0)
    game.make_move(move)
    move = Move(True, 10)
    game.make_move(move)

    expected_output = '3 3 0 3 3 1 4 1\n0 3 0 1 1 0 2 1\n----------\n1 1 1 1 0 0 3 0\n0 3 0 3 3 0 3 3\nBOT\n'
    assert str(game) == expected_output


def test_get_valid_moves():
    game = Game()
    move = Move(True, 0)
    game.make_move(move)
    move = Move(False, 0)
    game.make_move(move)
    move = Move(True, 10)
    game.make_move(move)
    valid_moves = game.get_valid_moves()

    np.testing.assert_equal(valid_moves, [1, 3, 4, 6, 7, 9])


def test_get_valid_moves_no_valid_moves():
    game = Game()
    game.top_state.state = np.ones((16))
    game_result = game.get_game_result()

    assert game_result == False


def test_get_valid_moves_empty_front():
    game = Game()
    empty_front_state = np.ones((16)) * 10
    empty_front_state[8:] = 0
    game.bottom_state.state = empty_front_state
    game_result = game.get_game_result()

    assert game_result == True


def test_get_valid_moves_default():
    game = Game()
    game_result = game.get_game_result()

    assert game_result is None
