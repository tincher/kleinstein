

class Move(object):

    __slots__ = ["top_turn", "field"]

    def __init__(self, top_turn, field):
        """Init a new move

        Parameters
        ----------
        top : bool
            whether the move is for the top or bottom player
        field : int
            which field the move starts with (counting from bottom-left from player-view in play-direction)
        """
        self.top_turn = top_turn
        self.field = field
