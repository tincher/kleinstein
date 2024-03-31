import random
from argparse import ArgumentParser

import mlflow
import numpy as np
import torch
import yaml
from tqdm import tqdm, trange

from engines import EngineFactory
from engines.td_engine import predict_best_move
from game_src.game import Game
from game_src.game_environment import GemStoneEnv
from game_src.move import Move
from td_stones.learning_rate_manager import LearningRateManager
from td_stones.model import TDStones


def main(training_config):
    mlflow.log_params(training_config, False)

    model, games_played, evaluations = train_model(training_config)

    log_model(model, games_played)
    return 1 - np.max(evaluations)


def train_model(training_config):
    model = TDStones(input_units=32, hidden_units=training_config["architecture"]["hidden_units"])
    learning_rate_manager = LearningRateManager(training_config["training"]["learning_rate"])

    env = GemStoneEnv()

    for games_played in trange(training_config["training"]["epochs"], leave=True, position=0):
        evaluations = []
        previous_prediction = None
        previous_second_term = [np.zeros(param.shape) for param in model.parameters()]
        learning_rate = learning_rate_manager.get_learning_rate()
        mlflow.log_metric("learning rate", learning_rate, synchronous=False)

        total_difference = 0
        total_abs_difference = 0
        steps = 0

        while not env.terminated:
            with torch.no_grad():
                # next prediction
                (best_move, _, next_prediction) = predict_best_move(env.game, model)

            # calculate and store the grads for the output
            model.zero_grad()
            # current prediction
            prediction = model(torch.tensor(env.observation, dtype=torch.float))
            # check if this is correct
            prediction.backward()

            _, reward, _, _, _ = env.step(best_move)
            if reward:
                next_prediction = torch.tensor(reward)

            # current gradients
            current_gradients = [param.grad for param in model.parameters()]
            if previous_prediction is not None:
                prediction_difference = next_prediction - prediction.detach()
                total_difference += prediction_difference
                total_abs_difference += abs(prediction_difference)

                model, previous_second_term = td_learn(model, training_config["training"]["discount"], learning_rate,
                                                       current_gradients, previous_second_term, prediction_difference)

            previous_prediction = prediction.detach()

            mlflow.log_metric(f"prediction_{games_played}", total_difference, synchronous=False, step=steps)
            steps += 1

        env.reset()

        mlflow.log_metric("total difference", total_difference, synchronous=False, step=games_played)
        mlflow.log_metric("total abs difference", total_abs_difference, synchronous=False, step=games_played)
        if (games_played + 1) % 10 == 0:
            log_model_performance(model, games_played, evaluations)
    return model, games_played, evaluations


def td_learn(model, discount, learning_rate, gradients, previous_second_term, prediction_difference):
    second_term = previous_second_term.copy()
    for i in range(len(previous_second_term)):
        second_term[i] = gradients[i] + discount * previous_second_term[i]
    first_part = learning_rate * prediction_difference
    weight_change = [first_part * new_gradient for new_gradient in second_term]

    state_dict = model.state_dict().copy()
    for i, weight_key in enumerate(state_dict):
        state_dict[weight_key] = state_dict[weight_key] + weight_change[i]
    model.load_state_dict(state_dict)
    return model, second_term


def log_model_performance(model, games_played, evaluations):
    with torch.no_grad():
        count_win_rate = eval_model(10, model, "count")
        random_win_rate = eval_model(10, model, "random")
        evaluations.append(count_win_rate)
        mlflow.log_metric("vs count win rate", count_win_rate, games_played)
        mlflow.log_metric("vs random win rate", random_win_rate, games_played)

        torch.save(model.state_dict(), f"./models/{games_played}.pt")


def log_model(model, games_played):
    game_representation = np.stack((Game().top_state.state, Game().bottom_state.state)).reshape(-1)
    return mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=f"tdstones_{games_played}",
        registered_model_name=f"trained_{games_played}",
        input_example=game_representation
    )


def eval_model(game_count, top_model, enemy_type):
    bottom_engine = EngineFactory().get_engine(enemy_type)()
    top_engine = EngineFactory().get_engine("td")()
    top_engine.model = top_model

    game_results = {"eval_top": 0, "eval_random": 0}
    env = GemStoneEnv()

    for _ in trange(game_count, position=1, leave=False):
        while not env.terminated:
            current_engine = top_engine if env.game.top_turn else bottom_engine
            env.step(current_engine.find_move(env.game))

        if env.reward > 0:
            game_results["eval_top"] += 1
        else:
            game_results["eval_random"] += 1

        env.reset()

    top_win_rate = game_results["eval_top"] / game_count

    tqdm.write(f"win rate: {top_win_rate}")
    return top_win_rate


def read_training_config(training_config_path):
    with open(training_config_path, 'r') as file:
        return yaml.safe_load(file)


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--training_config", type=str)
    args, unknown = parser.parse_known_args()

    config = read_training_config(args.training_config)
    if config["random_seed"] is None:
        config["random_seed"] = random.random()
    set_random_seeds(config["random_seed"])

    mlflow.set_tracking_uri(uri=config["logging"]["tracking_url"])
    mlflow.set_experiment(config["logging"]["experiment_name"])
    with mlflow.start_run():
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("project", "kleinstein")
        main(config)
