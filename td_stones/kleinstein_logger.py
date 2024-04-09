from types import TracebackType

import mlflow

from game_src.game_environment import GemStoneEnv
from td_stones.td_eval import Evaluation


class KleinsteinLogger:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.run = None
        mlflow.set_tracking_uri(uri=self.config["logging"]["tracking_url"])

    def __enter__(self) -> None:
        self.run = mlflow.start_run()
        mlflow.set_experiment(self.config["logging"]["experiment_name"])
        mlflow.set_tag("project", "kleinstein")
        mlflow.log_params(self.config, synchronous=False)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        mlflow.end_run()

    def log_prediction_difference(self, name: str, difference: float, step: int) -> None:
        if self.config["logging"]["verbose"]:
            mlflow.log_metric(name, difference, synchronous=False, step=step)

    def log_evaluations(self, evaluations: list[Evaluation], step: int) -> None:
        for evaluation in evaluations:
            mlflow.log_metric(evaluation.enemy, evaluation.win_rate, step=step)

    def log_learning_rate(self, learning_rate: float, step: int) -> None:
        mlflow.log_metric("learning rate", learning_rate, step=step, synchronous=False)

    def log_model(self, games_played: int) -> None:
        game_representation = GemStoneEnv().observation
        return mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path=f"tdstones_{games_played}",
            registered_model_name=f"trained_{games_played}",
            input_example=game_representation,
        )
