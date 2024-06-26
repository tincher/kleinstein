from argparse import ArgumentParser

import scipy.optimize
import skopt
import torch
from skopt import callbacks
from tqdm import tqdm

from td_stones.td_train import main as td_stones_train


def optimize(game_count: int, checkpoint_path: str = "./checkpoint.pkl") -> None:
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(default_device)

    space = []
    space.append((0.01, 0.99))
    space.append((0.01, 0.99))
    space.append((1, 512))

    checkpoint_callback = callbacks.CheckpointSaver(checkpoint_path, store_objective=False)

    def optimization_function(hyper_parameters: tuple[float, float, float]) -> float:
        (discount, alpha, hidden_units) = hyper_parameters
        tqdm.write(f"game_count: {game_count}, discount: {discount}, alpha: {alpha}, hidden_units: {hidden_units}")
        return td_stones_train(game_count, discount, alpha, int(hidden_units))

    res = scipy.optimize.minimize(optimization_function, bounds=space, callback=checkpoint_callback, x0=(0.5, 0.5, 256))
    print(res.fun, print(res.x))  # default: 0.5 0.5 256 -> 0.5 0.5 256
    skopt.dump(res, "./final.pkl", store_objective=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--game_count", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args, unknown = parser.parse_known_args()
    optimize(args.iterations, args.game_count, args.checkpoint_path)
