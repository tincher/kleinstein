from skopt import gp_minimize, callbacks
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer
from td_stones.td_train import main
import skopt
import scipy.optimize
from argparse import ArgumentParser
from tqdm import tqdm
import torch


def optimize(iterations, game_count, checkpoint_path):
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(default_device)

    space = []
    space.append((0.01, 0.99))
    space.append((0.01, 0.99))
    space.append((1, 512))

    checkpoint_callback = callbacks.CheckpointSaver("./checkpoint.pkl", store_objective=False)

    def optimization_function(hyper_parameters):
        (discount, alpha, hidden_units) = hyper_parameters
        tqdm.write(f"game_count: {game_count}, discount: {discount}, alpha: {alpha}, hidden_units: {hidden_units}")
        return main(game_count, discount, alpha, int(hidden_units))

    previous_x0 = []
    previous_y0 = []

    res = scipy.optimize.minimize(optimization_function, bounds=space, callback=checkpoint_callback, x0=(0.5, 0.5, 256))
    print(res.fun, print(res.x))  # default: 0.5 0.5 256 -> 0.5 0.5 256
    skopt.dump(res, "./final.pkl", store_objective=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--game_count", type=int, default=100)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args, unknown = parser.parse_known_args()
    optimize(args.iterations, args.game_count, args.checkpoint_path)
