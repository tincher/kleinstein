from dataclasses import dataclass

from tqdm import trange

from engines.engine_factory import EngineFactory
from game_src.game_environment import GemStoneEnv
from td_stones.model import TDStones


@dataclass
class Evaluation:
    enemy: str
    win_rate: float


class ModelEvaluator:
    def __init__(self, config: dict) -> Evaluation:
        self.game_count = config["evaluation"]["games"]
        self.enemies = config["evaluation"]["enemies"]

    def evaluate(self, model: TDStones) -> list[Evaluation]:
        evaluations = []
        for enemy in self.enemies:
            bottom_engine = EngineFactory().get_engine(enemy)()
            top_engine = EngineFactory().get_engine("td")()
            top_engine.model = model

            game_results = {"eval_top": 0, "eval_random": 0}
            env = GemStoneEnv()

            for _ in trange(self.game_count, position=1, leave=False):
                while not env.terminated:
                    current_engine = top_engine if env.game.top_turn else bottom_engine
                    env.step(current_engine.find_move(env.game))

                if env.reward > 0:
                    game_results["eval_top"] += 1
                else:
                    game_results["eval_random"] += 1

                env.reset()

            evaluations.append(Evaluation(enemy, game_results["eval_top"] / self.game_count))
        return evaluations
