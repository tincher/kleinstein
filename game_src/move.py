from game_src.state import Turn


class Move:
    """Encodes a single player's move, e.g. a pit from which to take the stones and from which side."""

    __slots__ = ["top_turn", "field"]

    def __init__(self, top_turn: Turn, field: int) -> None:
        """Initialize a new move.

        Parameters
        ----------
        top_turn : bool
            whether the move is for the top or bottom player
        field : int
            which field the move starts with (counting from bottom-left from player-view in play-direction)
        """
        self.top_turn = top_turn
        self.field = field
