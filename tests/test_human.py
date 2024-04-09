from argparse import ArgumentParser

from engines import EngineFactory
from engines.base_engine import BaseEngine
from game_src.game import Game
from game_src.move import Move


def main(engine_name: str) -> None:
    game = Game()
    engine = EngineFactory().get_engine(engine_name)()

    # setup
    human_turn = int(input("Will you play as top?"))

    while (game_result := game.get_game_result()) is None:
        if game.top_turn == human_turn:
            move = Move(game.top_turn, get_human_move(game))
        else:
            print(game)
            move = Move(game.top_turn, get_engine_move(game, engine))
        game.make_move(move)

    if game_result == human_turn:
        print("You won, humans rule")
    else:
        print("You lost, machines are taking over")


def get_engine_move(game: Game, engine: BaseEngine) -> Move:
    return engine.find_move(game)


def get_human_move(game: Game) -> int:
    valid_moves = game.get_valid_moves()
    print(game)
    print(f"Choose your move: {valid_moves}")
    if not game.top_turn:
        print("+---+---+---+---+---+---+---+---+")
        print("| 15| 14| 13| 12| 11| 10| 9 | 8 |")
        print("+---+---+---+---+---+---+---+---+")
        print("| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |")
        print("+---+---+---+---+---+---+---+---+")
    else:
        print("+---+---+---+---+---+---+---+---+")
        print("| 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |")
        print("+---+---+---+---+---+---+---+---+")
        print("| 8 | 9 | 10| 11| 12| 13| 14| 15|")
        print("+---+---+---+---+---+---+---+---+")
    user_input = int(input())
    if user_input not in valid_moves:
        error_message = "Wrong input"
        raise ValueError(error_message)
    return user_input


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--engine", type=str, default="random")
    args = argparse.parse_args()
    main(args.engine)
