import torch


class TDStones(torch.nn.Module):
    """TDStones model, uses board as input and outputs who is winning.

    MLP that uses the board as input directly and predicts the outcome (-1, 1) whether bot (-1) or
    top (1) is winning in this position.
    """

    def __init__(self, hidden_units: int, input_units: int = 40) -> None:
        """Initialize an MLP to evaluate the board.

        Parameters
        ----------
        hidden_units : int
            number of hidden units in the MLP
        input_units : int, optional
            number of input units, by default 40
        """
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_units, hidden_units, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the winning probabilites for the given board.

        Parameters
        ----------
        x : torch.Tensor
            board representation

        Returns
        -------
        torch.Tensor
            calculated output
        """
        return self.linear_relu_stack(x)
