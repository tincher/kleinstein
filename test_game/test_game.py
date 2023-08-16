from game.main import Game, Move, State


def test_init_print():
    game = Game()
    expected_output = '2. 2. 2. 2. 2. 2. 2. 2.\n2. 2. 2. 2. 0. 0. 0. 0.\n-----------------------\n0. 0. 0. 0. 2. 2. 2. 2.\n2. 2. 2. 2. 2. 2. 2. 2.\nTOP'
    assert str(game) == expected_output


def test_move():
    state = State(top=True)
    move = Move(True, 11)
    last_field = state.make_move(move.field)
    assert state.get_representation() == ['[2 2 2 2 2 2 2 2]', '[2 2 2 0 1 1 0 0]']  # noqa
    assert last_field == 13
