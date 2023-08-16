from game.main import Game, Move, State


def test_init_print():
    game = Game()
    expected_output = '2 2 2 2 2 2 2 2\n2 2 2 2 0 0 0 0\n----------\n0 0 0 0 2 2 2 2\n2 2 2 2 2 2 2 2\nTOP\n'
    assert str(game) == expected_output


def test_move():
    state = State(top=True)
    move = Move(True, 11)
    last_field = state.make_move(move.field)
    assert state.get_representation() == ['[2 2 2 2 2 2 2 2]', '[2 2 2 0 1 1 0 0]']  # noqa
    assert last_field == 13


def test_make_moves():
    game = Game()
    move = Move(True, 0)
    game.make_moves(move)

    expected_output = '3 3 0 3 3 0 3 0\n0 3 3 0 1 1 1 0\n----------\n0 0 0 0 2 2 2 2\n2 2 2 2 2 2 2 2\nBOT\n'
    assert str(game) == expected_output


def test_make_moves_spillover():
    game = Game()
    move = Move(True, 0)
    game.make_moves(move)
    move = Move(False, 0)
    game.make_moves(move)
    move = Move(True, 10)
    game.make_moves(move)
    move = Move(False, 10)
    game.make_moves(move)
    move = Move(True, 14)
    game.make_moves(move)
    move = Move(False, 14)
    game.make_moves(move)
    move = Move(True, 1)
    game.make_moves(move)

    expected_output = '4 4 1 0 4 1 0 1\n1 3 0 1 2 0 0 2\n----------\n2 0 0 2 1 0 3 0\n1 3 0 3 3 0 3 3\nBOT\n'
    assert str(game) == expected_output
