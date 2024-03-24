import copy
from argparse import ArgumentParser

import mlflow
import numpy as np
import torch
import yaml
from tqdm import tqdm, trange

from engines import EngineFactory
from game_src import Game, Move
from td_stones.model import TDStones


class LearningRateManager():
    def __init__(self, learning_rate_config):
        self.learning_rate = learning_rate_config["initial"]
        self.schedule_learning_rate = learning_rate_config["schedule_learning_rate"]
        self.learning_rate_divisor = learning_rate_config["learning_rate_divisor"]
        self.schedule_learning_rate_steps = learning_rate_config["schedule_learning_rate_steps"]

        self.steps_done = 0

    def get_learning_rate(self):
        if not self.schedule_learning_rate:
            return self.learning_rate
        self.steps_done += 1
        if self.steps_done >= self.schedule_learning_rate_steps:
            self.steps_done = 0
            self.learning_rate = self.learning_rate / self.learning_rate_divisor
        return self.learning_rate


def main(training_config):
    mlflow.log_params(training_config, False)

    model, games_played, evaluations = train_model(training_config)

    log_model(model, games_played)
    return 1 - np.max(evaluations)


def train_model(training_config):
    model = TDStones(input_units=32, hidden_units=training_config["architecture"]["hidden_units"])
    learning_rate_manager = LearningRateManager(training_config["training"]["learning_rate"])

    for games_played in trange(training_config["training"]["epochs"], leave=True, position=0):
        evaluations = []
        previous_prediction = None
        previous_second_term = [np.zeros(param.shape) for param in model.parameters()]
        learning_rate = learning_rate_manager.get_learning_rate()

        game = Game()

        while game.get_game_result() is None:
            with torch.no_grad():
                # next prediction
                (best_move, _, next_prediction) = predict_best_move(game, model)

            move = Move(game.top_turn, best_move)
            game.make_move(move)

            current_turn = game.top_turn
            own_state = game.top_state if current_turn else game.bottom_state
            enemy_state = game.top_state if not current_turn else game.bottom_state

            if (result := game.get_game_result()) is not None:
                next_prediction = torch.tensor(int(result))

            game_representation = torch.tensor(np.stack((own_state.state, enemy_state.state)),
                                               dtype=torch.float).reshape((-1))

            # calculate and store the grads for the output
            model.zero_grad()
            # current prediction
            prediction = model(game_representation)
            # check if this is correct
            prediction.backward()

            # current gradients
            current_gradients = [param.grad for param in model.parameters()]
            if previous_prediction is not None:
                prediction_difference = next_prediction - prediction.detach()
                model, previous_second_term = td_learn(model, training_config["training"]["discount"], learning_rate,
                                                       current_gradients, previous_second_term, prediction_difference)

            previous_prediction = prediction.detach()

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

    for _ in trange(game_count, position=1, leave=False):
        game = Game()
        while (game_result := game.get_game_result()) is None:
            current_engine = top_engine if game.top_turn else bottom_engine
            move = Move(game.top_turn, current_engine.find_move(game))
            game.make_move(move)

        if game_result:
            game_results["eval_top"] += 1
        else:
            game_results["eval_random"] += 1

    top_win_rate = game_results["eval_top"] / game_count

    tqdm.write(f"win rate: {top_win_rate}")
    return top_win_rate


def predict_best_move(game, model):
    valid_moves = game.get_valid_moves()

    best_output, best_output_index, best_game_representation = -1, -1, None

    current_turn = game.top_turn

    for move_index, move in enumerate(valid_moves):

        current_game = copy.deepcopy(game)

        current_game.make_move(Move(current_turn, move))

        own_state = current_game.top_state if current_turn else current_game.bottom_state
        enemy_state = current_game.top_state if not current_turn else current_game.bottom_state

        game_representation = torch.tensor(np.stack((own_state.state, enemy_state.state)),
                                           dtype=torch.float).reshape((-1))
        model_output = model(game_representation)

        if (own_win_probability := model_output[0]) > best_output:
            best_output = own_win_probability
            best_output_index = move_index
            best_game_representation = game_representation

    return valid_moves[best_output_index], best_game_representation, best_output


def read_training_config(training_config_path):
    with open(training_config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--training_config", type=str)
    args, unknown = parser.parse_known_args()

    config = read_training_config(args.training_config)

    mlflow.set_tracking_uri(uri=config["logging"]["tracking_url"])
    mlflow.set_experiment(config["logging"]["experiment_name"])
    with mlflow.start_run(config["logging"]["run_name"]):
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("project", "kleinstein")
        main(config)
