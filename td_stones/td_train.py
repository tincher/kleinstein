import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import trange

from engines.td_engine import TDEngine
from game_src.game_environment import GemStoneEnv
from td_stones.kleinstein_logger import KleinsteinLogger
from td_stones.learning_rate_manager import LearningRateManager
from td_stones.model import TDStones
from td_stones.td_eval import ModelEvaluator


class ModelTrainer:
    def __init__(self, training_config: dict, logger: KleinsteinLogger) -> None:
        self.config = training_config
        self.model = TDStones(input_units=32, hidden_units=self.config["architecture"]["hidden_units"])
        self.learning_rate_manager = LearningRateManager(self.config["training"]["learning_rate"])
        self.logger = logger

    def train_model(self) -> tuple[TDStones, int, list[float]]:
        env = GemStoneEnv()

        for games_played in trange(self.config["training"]["epochs"], leave=True, position=0):
            learning_rate = self.learning_rate_manager.get_learning_rate()
            self.logger.log_learning_rate(learning_rate, step=games_played)

            evaluations = []
            previous_prediction = None
            prediction_difference = None
            previous_second_term = [np.zeros(param.shape) for param in self.model.parameters()]

            total_difference = 0
            total_abs_difference = 0
            steps = 0

            while not env.terminated:
                with torch.no_grad():
                    # next prediction
                    (best_move, next_prediction) = TDEngine.predict_best_move(env.game, self.model)

                # calculate and store the grads for the output
                self.model.zero_grad()
                # current prediction
                prediction = self.model(torch.tensor(env.observation, dtype=torch.float))
                # check if this is correct
                prediction.backward()

                _, reward, _, _, _ = env.step(best_move)
                if reward:
                    next_prediction = torch.tensor(reward)

                # current gradients
                current_gradients = [param.grad for param in self.model.parameters()]
                if previous_prediction is not None:
                    prediction_difference = next_prediction - prediction.detach()
                    total_difference += prediction_difference
                    total_abs_difference += abs(prediction_difference)

                    model, previous_second_term = self.td_learn(
                        learning_rate,
                        current_gradients,
                        previous_second_term,
                        prediction_difference,
                    )

                previous_prediction = prediction.detach()
                self.logger.log_prediction_difference(
                    f"predictions_{games_played}", prediction_difference, step=env.steps
                )
                steps += 1

            env.reset()
            self.logger.log_prediction_difference("total difference", total_difference, games_played)
            self.logger.log_prediction_difference("total abs difference", total_abs_difference, games_played)
            if (games_played + 1) % 10 == 0:
                evaluations = ModelEvaluator(self.config).evaluate(model)
                self.logger.log_evaluations(evaluations, games_played)

        return model, games_played, evaluations

    def td_learn(
        self,
        learning_rate: float,
        gradients: list[torch.Tensor],
        previous_second_term: list[torch.Tensor],
        prediction_difference: float,
    ) -> tuple[TDStones, list[torch.Tensor]]:
        second_term = previous_second_term.copy()
        for i in range(len(previous_second_term)):
            second_term[i] = gradients[i] + self.config["training"]["discount"] * previous_second_term[i]
        first_part = learning_rate * prediction_difference
        weight_change = [first_part * new_gradient for new_gradient in second_term]

        state_dict = self.model.state_dict().copy()
        for i, weight_key in enumerate(state_dict):
            state_dict[weight_key] = state_dict[weight_key] + weight_change[i]
        self.model.load_state_dict(state_dict)
        return self.model, second_term


def read_training_config(training_config_path: str) -> dict:
    with Path(training_config_path).open() as file:
        return yaml.safe_load(file)


def set_random_seeds(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)  # noqa: NPY002 , this is just in case a library uses it
    random.seed(random_seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_config", type=str)
    args, unknown = parser.parse_known_args()

    config = read_training_config(args.training_config)
    if config["random_seed"] is None:
        config["random_seed"] = np.random.default_rng().random()
    set_random_seeds(config["random_seed"])

    with KleinsteinLogger(config) as logger:
        model_trainer = ModelTrainer(config, logger)
        model_trainer.train_model()
