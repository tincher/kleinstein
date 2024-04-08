from typing import Any, ClassVar, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import Game, Move


class GemStoneEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(self, render_mode: str | None = None) -> None:
        self.observation_space = spaces.MultiDiscrete([32 * 2] * 32)
        self.terminated = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = MoveSpace(Game())

    @property
    def game(self) -> Game:
        return self.action_space.game

    @game.setter
    def game(self, game: Game) -> None:
        self.action_space.game = game

    @property
    def observation(self) -> np.ndarray:
        return np.stack((self.game.top_state.state, self.game.bottom_state.state)).reshape(-1)

    @property
    def reward(self) -> int:
        reward = self.game.get_game_result()
        if reward is None:
            self.terminated = False
            reward = 0
        else:
            self.terminated = True
        return reward

    def step(self, action: np.ndarray) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        valid_moves = self.game.get_valid_moves()
        assert action in valid_moves

        move = Move(self.game.top_turn, action)
        self.game.make_move(move)
        info = {"turn": self.game.top_turn, "valid": valid_moves}

        return self.observation, self.reward, self.terminated, False, info

    def reset(self, **_: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self.game = Game()
        self.terminated = False
        return self.observation, {}


class MoveSpace(spaces.Space):
    game: Game

    def __init__(self, game: Game) -> None:
        self.game = game

    def sample(self) -> Move:
        return np.random.default_rng().choice(self.game.get_valid_moves())
