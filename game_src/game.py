import numpy as np

from .move import Move
from .state import State

TOP = True
BOT = False
MAX_MOVE_COUNT = 600
MIN_STONE_COUNT = 2


class Game:
    def __init__(self) -> None:
        self.top_state = State(TOP)
        self.bottom_state = State(BOT)
        self.top_turn = TOP
        self.move_count = 0

    def get_valid_moves(self) -> np.ndarray:
        active_state = self.top_state if self.top_turn else self.bottom_state
        return np.where(active_state.state > 1)[0]

    def make_move(self, move: Move) -> None:
        if move.top_turn != self.top_turn:
            print("WARNING: move turn is different from game turn")
        active_state, passive_state = self.top_state, self.bottom_state
        if not self.top_turn:
            active_state, passive_state = passive_state, active_state

        move_field = move.field
        player_move = True
        current_count = self.get_stone_count(move_field, active_state, player_move=player_move)

        while current_count > 1:
            move_field = self.execute_single_move(move_field, active_state, passive_state, player_move=player_move)
            player_move = False
            current_count = self.get_stone_count(move_field, active_state, player_move=player_move)

            if self.get_game_result() is not None:
                break

        self.top_turn = not self.top_turn
        self.move_count += 1

    def execute_single_move(
        self, move_field: int, active_state: np.ndarray, passive_state: np.ndarray, *, player_move: bool
    ) -> int:
        """Execute a single move, i.e. take x stones, fill x subsequent pits, returns field + x.

        Parameters
        ----------
        move_field : int
            _description_
        active_state : np.ndarray
            _description_
        passive_state : np.ndarray
            _description_
        player_move: bool
            _description_, default = False

        Returns
        -------
        int
            last field where a stone was put into
        """
        own_stone_count = active_state[move_field]
        enemy_stone_count = 0
        start_index_for_being_able_to_steal = 7
        if move_field > start_index_for_being_able_to_steal and not player_move:
            opposing_field_index = (15 - (move_field - 8)) % 16
            enemy_stone_count = passive_state[opposing_field_index]
            passive_state[opposing_field_index] = 0
        stone_count = own_stone_count + enemy_stone_count

        last_field = (move_field + stone_count) % 16
        active_state[move_field] = 0
        active_state[move_field + 1 : move_field + stone_count + 1] += 1
        if last_field < move_field:
            active_state[: last_field + 1] += 1
        return last_field

    def get_stone_count(self, move: int, active_state: np.ndarray, *, player_move: bool) -> int:
        if (current_count := active_state[move]) < MIN_STONE_COUNT and player_move:
            error_message = (
                f"Invalid Move: There are not enough stones in the pit!pit: {move}, entry: {active_state[move]}"
            )
            raise ValueError(error_message)

        return current_count

    def get_game_result(self) -> int | None:
        if self.top_state.is_lost():
            return -1
        if self.bottom_state.is_lost():
            return 1
        if self.move_count > MAX_MOVE_COUNT:
            return 0
        return None

    def __str__(self) -> str:
        result = []
        result.extend(self.top_state.get_representation())
        result.extend(self.bottom_state.get_representation())

        result = [entry.replace("[", "").replace("]", "").strip() for entry in result]
        result.insert(2, "-" * 10)
        result.append("TOP" if self.top_turn else "BOT")
        return "\n".join(result) + "\n"

    def __repr__(self) -> str:
        return str(self)
