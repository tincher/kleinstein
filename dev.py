from game_src import Game, Move


def main():
    game = Game()
    print(game)
    move = Move(True, 0)
    game.make_move(move)
    print(game)
    move = Move(False, 0)
    game.make_move(move)
    print(game)
    move = Move(True, 10)
    game.make_move(move)
    print(game)


if __name__ == "__main__":
    main()
