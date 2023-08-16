from game.main import Game, Move, generate_valid_moves


def main():
    game = Game()
    print(game)
    move = Move(True, 0)
    game.make_moves(move)
    print(game)
    move = Move(False, 0)
    game.make_moves(move)
    print(game)
    move = Move(True, 10)
    game.make_moves(move)
    print(game)
    move = Move(False, 10)
    game.make_moves(move)
    print(game)
    move = Move(True, 14)
    game.make_moves(move)
    print(game)
    move = Move(False, 14)
    game.make_moves(move)
    print(game)
    move = Move(True, 1)
    game.make_moves(move)
    print(game)


if __name__ == "__main__":
    main()
