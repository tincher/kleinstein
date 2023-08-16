from game.main import Game, Move, generate_valid_moves


def main():
    game = Game()
    print(game)
    move = Move(True, 11)
    last_field = game.top_state.make_move(move.field)
    print(game)
    print(last_field)


if __name__ == "__main__":
    main()
