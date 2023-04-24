from typing import List, Tuple

import numpy as np
import flwr as fl

from flwr.common import Metrics
from vowpalwabbit.sklearn import VW, VWClassifier, VWMultiClassifier, tovw

import client
import gen_data
import utils
from strategy import WeightedFedAvg


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

def init_model(num_features, num_classes):
    dummy_X = ' '.join([f"{j}:0.0" for j in range(num_features)])
    dummy_y = ' '.join([f"{j+1}:{0}" for j in range(num_classes)])
    dummy_data = [f"{dummy_y} 1 | {dummy_X}" for _ in range(2)]
    model = VW(csoaa=num_classes,
               convert_to_vw=False,
               convert_labels=False,
               passes=1)
    model.fit(dummy_data)
    return model

def create_client_fn(data, aux, opt_model, num_features, num_classes):
    def client_fn(cid: str) -> client.VowpalWabbitClient:
        client_data, client_aux = data[int(cid)], aux[int(cid)]
        model = init_model(num_features, num_classes)
        return client.VowpalWabbitClient(cid, model, opt_model, client_data, client_aux)
    return client_fn

def create_evaluate_metrics_aggregation_fn(weights):
    def weighted_average_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        opt_reward = [weights[i] * metric["opt_reward"] for i, (_, metric) in enumerate(metrics)]
        reward = [weights[i] * metric["reward"] for i, (_, metric) in enumerate(metrics)]
        regret = [weights[i] * metric["regret"] for i, (_, metric) in enumerate(metrics)]
        return {"opt_reward": sum(opt_reward), "reward": sum(reward), "regret": sum(regret)}
    return weighted_average_aggregation_fn

def run_federated_learning(data, aux, opt_model,
                           num_features, num_classes,
                           num_rounds=5, num_clients=5, client_weights=None):
    # Create client function
    client_fn = create_client_fn(data, aux, opt_model, num_features, num_classes)

    # Define strategy
    if client_weights is None:
        client_weights = np.ones(num_clients) / num_clients
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=create_evaluate_metrics_aggregation_fn(client_weights),
        )

    # Start simulation
    sim = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    return sim


if __name__ == "__main__":
    np.random.seed(42)

    # Federated config
    NUM_ROUNDS = 3
    NUM_CLIENTS = 4
    NUM_ACTIONS = 4
    NUM_FEATURES = 10
    SAMPLE_SIZES = np.array([1, 1, 1, 1]) * 100000
    SAMPLE_SIZES[0] = 500
    CLIENT_WEIGHTS = SAMPLE_SIZES / np.sum(SAMPLE_SIZES)
    
    # Generate data
    data, aux = gen_data.generate_observational_data(num_clients=NUM_CLIENTS,
                                                    sample_sizes=SAMPLE_SIZES,
                                                    num_features=NUM_FEATURES,
                                                    num_actions=NUM_ACTIONS)
    costs = {client_id: aux[client_id]["costs"] for client_id in range(NUM_CLIENTS)}    

    # Run federated learning
    global_vw = run_federated_learning(data, costs,
                                       num_features=NUM_FEATURES,
                                       num_classes=NUM_ACTIONS,
                                       num_rounds=NUM_ROUNDS,
                                       num_clients=NUM_CLIENTS,
                                       client_weights=CLIENT_WEIGHTS)

    # Train on all data
    agg_X = []
    agg_A = []
    agg_Y = []
    agg_costs = []
    for client_id in range(NUM_CLIENTS):
        agg_X.extend(aux[client_id]["X"])
        agg_A.extend(aux[client_id]["A"])
        agg_Y.extend(aux[client_id]["Y"])
        agg_costs.extend(aux[client_id]["costs"])
    agg_X = np.array(agg_X)
    agg_A = np.array(agg_A)
    agg_Y = np.array(agg_Y)
    agg_costs = np.array(agg_costs)

    # Compute AIPW scores
    crossfit_map, mu, e = utils.cross_fit_nuisance_params(agg_X, agg_A, agg_Y, NUM_ACTIONS)
    agg_costs = -utils.compute_AIPW_scores(agg_X, agg_A, agg_Y, NUM_ACTIONS, crossfit_map, mu, e)
    agg_data = utils.to_vw_format(agg_X, agg_A, agg_costs)

    model = VW(csoaa=NUM_ACTIONS,
               convert_to_vw=False,
               convert_labels=False,
               passes=3)
    model.fit(agg_data)

    y_pred = model.predict(agg_data)
    opt_reward, reward, regret = utils.compute_regret(y_pred, agg_costs)
    # print(model.get_coefs())
    # print(data[client_id])
    # print(y_pred[-100:])
    print(f"Aggregate: opt_reward={opt_reward}, reward={reward}, regret={regret}")

    y_pred = np.random.choice(NUM_ACTIONS, size=len(agg_data))
    opt_reward, reward, regret = utils.compute_regret(y_pred, agg_costs)
    print(f"Random Aggregate: opt_reward={opt_reward}, reward={reward}, regret={regret}")


    # Train on local data
    client_id = 0
    model = VW(csoaa=NUM_ACTIONS,
               convert_to_vw=False,
               convert_labels=False,
               passes=3)
    model.fit(data[client_id])

    y_pred = model.predict(data[client_id])
    opt_reward, reward, regret = utils.compute_regret(y_pred, costs[client_id])
    print(f"Local Client {client_id}: opt_reward={opt_reward}, reward={reward}, regret={regret}")


    y_pred = np.random.choice(NUM_ACTIONS, size=len(data[client_id]))
    opt_reward, reward, regret = utils.compute_regret(y_pred, costs[client_id])
    print(f"Random Local Client {client_id}: opt_reward={opt_reward}, reward={reward}, regret={regret}")