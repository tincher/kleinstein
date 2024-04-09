from argparse import ArgumentParser

from test_human import get_engine_move
from tqdm import trange

from engines import EngineFactory
from game_src.game import Game
from game_src.move import Move


def main(top_engine_name: str, bottom_engine_name: str, game_count: int) -> None:
    top_engine = EngineFactory().get_engine(top_engine_name)()
    bottom_engine = EngineFactory().get_engine(bottom_engine_name)()

    game_results = {f"{top_engine_name}_top": 0, f"{bottom_engine_name}_bottom": 0}

    for _ in trange(game_count):
        game = Game()
        while (game_result := game.get_game_result()) is None:
            current_engine = top_engine if game.top_turn else bottom_engine
            move = Move(game.top_turn, get_engine_move(game, current_engine))
            game.make_move(move)

        if game_result:
            game_results[f"{top_engine_name}_top"] += 1
        else:
            game_results[f"{bottom_engine_name}_bottom"] += 1

    top_win_rate = game_results[f"{top_engine_name}_top"] / game_count
    bottom_win_rate = game_results[f"{bottom_engine_name}_bottom"] / game_count

    print(game_results)
    print(f"top {top_engine_name} win rate: {top_win_rate}")
    print(f"bottom {bottom_engine_name} win rate: {bottom_win_rate}")


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--top_engine", type=str)
    argparse.add_argument("--bottom_engine", type=str)
    argparse.add_argument("--game_count", type=int, default=1000)
    args = argparse.parse_args()
    main(args.top_engine, args.bottom_engine, args.game_count)
