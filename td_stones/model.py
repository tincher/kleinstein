from torch import nn


class TDStones(nn.Module):
    def __init__(self, hidden_units, input_units=40):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_units, hidden_units, bias=False),
            nn.ReLU(),
            # nn.Linear(hidden_units, hidden_units, bias=False),
            # nn.ReLU(),
            nn.Linear(hidden_units, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        probs = self.linear_relu_stack(x)
        return probs
