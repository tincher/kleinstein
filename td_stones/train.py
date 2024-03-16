from argparse import ArgumentParser
from td_stones.model import TDStones
from tqdm import trange, tqdm
from engines import EngineFactory
from game_src import Game, Move
import numpy as np
import copy
import torch
import mlflow


def main(game_count, discount, alpha, hidden_units):
    # Log the hyperparameters
    mlflow.log_param("game_count", game_count, False)
    mlflow.log_param("discount", discount, False)
    mlflow.log_param("alpha", alpha, False)
    mlflow.log_param("hidden_units", hidden_units, False)

    model = TDStones(input_units=32, hidden_units=hidden_units)
    gradients = []

    final_predictions = [[], []]

    for games_played in trange(game_count, leave=True, position=0):
        evaluations = []

        game = Game()

        while (game_result := game.get_game_result()) is None:

            with torch.no_grad():
                (best_move, best_game_representation,
                 final_predictions[int(not game.top_turn)]) = predict_best_move(game, model)

            move = Move(game.top_turn, best_move)
            game.make_move(move)

            # calculate and store the grads for the output
            model.zero_grad()
            prediction = model(best_game_representation)
            # check if this is correct
            prediction.backward(prediction)

            current_grads = [param.grad for param in model.parameters()]
            gradients.append(current_grads)

        result = int(game_result)

        model = learn(result, model, gradients, final_predictions, discount, alpha)

        if (games_played + 1) % 10 == 0:
            with torch.no_grad():
                top_win_rate = eval_model(50, model)
                evaluations.append(top_win_rate)
                mlflow.log_metric("bot win rate", top_win_rate, games_played)

                torch.save(model.state_dict(), f"./models/{games_played}.pt")

    game_representation = np.stack((Game().top_state.state, Game().bottom_state.state)).reshape(-1)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=f"tdstones_{games_played}",
        registered_model_name=f"trained_{games_played}",
        input_example=game_representation
    )
    return 1 - np.max(evaluations)


def eval_model(game_count, top_model):
    bottom_engine = EngineFactory().get_engine("random")()
    top_engine = EngineFactory().get_engine("td")()
    top_engine.model = top_model

    game_results = {"eval_top": 0, "eval_random": 0}

    for _ in trange(game_count, position=1, leave=False):
        game = Game()
        while (game_result := game.get_game_result()) is None:
            current_engine = top_engine if game.top_turn else bottom_engine
            move = Move(game.top_turn, get_engine_move(game, current_engine))
            game.make_move(move)

        if game_result:
            game_results["eval_top"] += 1
        else:
            game_results["eval_random"] += 1

    top_win_rate = game_results["eval_top"] / game_count

    tqdm.write(f"top win rate: {top_win_rate}")
    return top_win_rate


def get_engine_move(game, engine):
    return engine.find_move(game)


def learn(result, model, gradients, final_predictions, discount, alpha):
    discounted_gradients = [torch.zeros(grads.shape) for grads in gradients[0]]
    end_time_step = len(gradients)
    for time_step, time_step_gradients in enumerate(gradients):
        discounted_gradients = [previous + gradient * discount**(end_time_step - (time_step // 2))
                                for gradient, previous in zip(time_step_gradients, discounted_gradients)]

    # do the weight change
        final_reward_signal = torch.nn.functional.one_hot(torch.tensor(int(not result)), num_classes=2)
    top = alpha * (final_reward_signal[0] - final_predictions[0])
    bottom = alpha * (final_reward_signal[1] - final_predictions[1])

    state_dict = model.state_dict()
    for i, weight_key in enumerate(state_dict):
        direction = top if i % 2 == 0 else bottom
        state_dict[weight_key] = state_dict[weight_key] + discounted_gradients[i] * direction
    model.load_state_dict(state_dict)

    return model


def predict_best_move(game, model):
    valid_moves = game.get_valid_moves()

    best_output, best_output_index, best_game_representation = -1, -1, None

    current_turn = game.top_turn

    for move_index, move in enumerate(valid_moves):

        current_game = copy.deepcopy(game)

        current_game.make_move(Move(current_turn, move))

        own_state = current_game.top_state if current_turn else current_game.bottom_state
        enemy_state = current_game.top_state if not current_turn else current_game.bottom_state

        game_representation = torch.tensor(np.stack((own_state.state, enemy_state.state)),
                                           dtype=torch.float).reshape((-1))
        model_output = model(game_representation)

        if (own_win_probability := model_output[0]) > best_output:
            best_output = own_win_probability
            best_output_index = move_index
            best_game_representation = game_representation

    return valid_moves[best_output_index], best_game_representation, best_output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.1)  # can be seen as a learning rate
    parser.add_argument("--hidden_units", type=int, default=100)
    parser.add_argument("--game_count", type=int, default=100)
    args, unknown = parser.parse_known_args()

    mlflow.set_tracking_uri(uri="http://192.168.178.22:5000")
    mlflow.set_experiment("[Kleinstein] TD training")
    with mlflow.start_run():
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("project", "kleinstein")
        main(int(args.game_count), args.discount, args.alpha, int(args.hidden_units))
