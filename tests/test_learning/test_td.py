from td_stones.model import TDStones
from td_stones.td_train import td_learn


def test_td_learn():
    # arrange
    model = TDStones(1)
    gradients = []
    second_term = []
    for _ in model.state_dict():
        gradients.append(1)
        second_term.append(2)
    prediction_difference = 1
    learning_rate, discount = 1, 1

    # act
    model, _ = td_learn(model, discount, learning_rate, gradients, second_term, prediction_difference)

    # assert
