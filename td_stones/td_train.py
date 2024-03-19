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
    log_hyperparameters(game_count, discount, alpha, hidden_units)

    model, games_played, evaluations = train_model(game_count, discount, alpha, hidden_units)

    log_model(model, games_played)
    return 1 - np.max(evaluations)


def log_hyperparameters(game_count, discount, alpha, hidden_units):
    mlflow.log_param("game_count", game_count, False)
    mlflow.log_param("discount", discount, False)
    mlflow.log_param("alpha", alpha, False)
    mlflow.log_param("hidden_units", hidden_units, False)


def train_model(game_count, discount, alpha, hidden_units):
    model = TDStones(input_units=32, hidden_units=hidden_units)

    for games_played in trange(game_count, leave=True, position=0):
        evaluations = []
        previous_prediction = None
        previous_second_term = [np.zeros(param.shape) for param in model.parameters()]

        game = Game()

        while (game_result := game.get_game_result()) is None:
            with torch.no_grad():
                # nächste prediction
                (best_move, _, next_prediction) = predict_best_move(game, model)

            move = Move(game.top_turn, best_move)
            game.make_move(move)

            current_turn = game.top_turn
            own_state = game.top_state if current_turn else game.bottom_state
            enemy_state = game.top_state if not current_turn else game.bottom_state

            if (result := game.get_game_result()) is not None:
                next_prediction = torch.tensor(int(result == current_turn))

            game_representation = torch.tensor(np.stack((own_state.state, enemy_state.state)),
                                               dtype=torch.float).reshape((-1))

            # calculate and store the grads for the output
            model.zero_grad()
            # aktuelle prediction
            prediction = model(game_representation)
            # check if this is correct
            prediction.backward(prediction)

            # aktuelle gradients
            current_gradients = [param.grad for param in model.parameters()]
            if previous_prediction is not None:
                prediction_difference = next_prediction - prediction.detach()[int(not current_turn)]
                model, previous_second_term = td_learn(model, discount, alpha, current_gradients, previous_second_term,
                                                       prediction_difference)

            previous_prediction = prediction.detach()

        if (games_played + 1) % 10 == 0:
            log_model_performance(model, games_played, evaluations)
    return model, games_played, evaluations


def td_learn(model, discount, alpha, gradients, previous_second_term, prediction_difference):
    new_gradients = previous_second_term.copy()
    for i in range(len(previous_second_term)):
        new_gradients[i] = discount * (gradients[i] + discount * previous_second_term[i])
    first_part = alpha * prediction_difference
    weight_change = [first_part * new_gradient for new_gradient in new_gradients]

    state_dict = model.state_dict()
    for i, weight_key in enumerate(state_dict):
        state_dict[weight_key] = state_dict[weight_key] + weight_change[i]
    model.load_state_dict(state_dict)
    return model, weight_change


def log_model_performance(model, games_played, evaluations):
    with torch.no_grad():
        count_win_rate = eval_model(10, model, "count")
        random_win_rate = eval_model(10, model, "random")
        evaluations.append(count_win_rate)
        mlflow.log_metric("vs count win rate", count_win_rate, games_played)
        mlflow.log_metric("vs random win rate", random_win_rate, games_played)

        torch.save(model.state_dict(), f"./models/{games_played}.pt")


def log_model(model, games_played):
    game_representation = np.stack((Game().top_state.state, Game().bottom_state.state)).reshape(-1)
    return mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=f"tdstones_{games_played}",
        registered_model_name=f"trained_{games_played}",
        input_example=game_representation
    )


def eval_model(game_count, top_model, enemy_type):
    bottom_engine = EngineFactory().get_engine(enemy_type)()
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

    tqdm.write(f"win rate: {top_win_rate}")
    return top_win_rate


def get_engine_move(game, engine):
    return engine.find_move(game)


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
    mlflow.set_experiment("[Kleinstein] TD debug")
    with mlflow.start_run():
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("project", "kleinstein")
        main(int(args.game_count), args.discount, args.alpha, int(args.hidden_units))
