from argparse import ArgumentParser

import skopt
import torch
from skopt import callbacks, gp_minimize
from skopt.space.space import Integer, Real
from skopt.utils import use_named_args
from tqdm import tqdm

from td_stones.td_train import main as td_stones_train


def optimize(iterations: int, game_count: int, checkpoint_path: str) -> None:
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(default_device)
    print(torch.tensor([1.2, 3]).device)

    space = []
    space.append(Real(name="discount", prior="uniform", low=0.01, high=0.99))
    space.append(Real(name="alpha", prior="log-uniform", low=0.001, high=0.9))
    space.append(Integer(name="hidden_units", prior="log-uniform", low=1, high=512))

    checkpoint_callback = callbacks.CheckpointSaver("./result.pkl", store_objective=False)

    @use_named_args(space)
    def optimization_function(discount: float, alpha: float, hidden_units: int) -> float:
        tqdm.write(f"game_count: {game_count}, discount: {discount}, alpha: {alpha}, hidden_units: {hidden_units}")
        return td_stones_train(game_count, discount, alpha, hidden_units)

    previous_x0 = []
    previous_y0 = []
    if checkpoint_path is not None:
        checkpoint = skopt.load(checkpoint_path)
        previous_x0 = checkpoint.x_iters
        previous_y0 = checkpoint.func_vals

    res = gp_minimize(
        optimization_function,
        space,
        n_calls=iterations,
        n_jobs=5,
        random_state=1234,
        callback=checkpoint_callback,
        x0=previous_x0,
        y0=previous_y0,
    )
    print(res.fun, print(res.x))
    skopt.dump(res, "./final.pkl", store_objective=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--game_count", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args, unknown = parser.parse_known_args()
    optimize(args.iterations, args.game_count, args.checkpoint_path)
