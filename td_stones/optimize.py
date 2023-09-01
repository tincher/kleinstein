from skopt import gp_minimize, callbacks
from skopt.utils import use_named_args
from skopt.space.space import Real, Integer
from td_stones.train import main
import skopt
from argparse import ArgumentParser
from tqdm import tqdm
import torch


def optimize(iterations, game_count):
    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_default_device(default_device)
    print(torch.tensor([1.2, 3]).device)

    space = []
    space.append(Real(name="discount", prior="uniform", low=0.01, high=0.99))
    space.append(Real(name="alpha", prior="log-uniform", low=0.001, high=0.9))
    space.append(Integer(name="hidden_units", prior="log-uniform", low=1, high=512))

    checkpoint_callback = callbacks.CheckpointSaver("./result.pkl", store_objective=False)

    @use_named_args(space)
    def optimization_function(discount, alpha, hidden_units):
        tqdm.write(f"game_count: {game_count}, discount: {discount}, alpha: {alpha}, hidden_units: {hidden_units}")
        return main(game_count, discount, alpha, hidden_units)

    res = gp_minimize(optimization_function, space, n_calls=iterations, n_jobs=5, random_state=1234,
                      callback=checkpoint_callback)
    print(res.fun, print(res.x))
    skopt.dump(res, "./final.pkl", store_objective=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--game_count", type=int, default=100)
    args, unknown = parser.parse_known_args()
    optimize(args.iterations, args.game_count)
