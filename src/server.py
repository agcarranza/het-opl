from typing import List, Tuple

import numpy as np
import flwr as fl

from flwr.common import Metrics
from vowpalwabbit.sklearn import VW

import client


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

def create_client_fn(data, aux, opt_global_model, opt_local_models, num_features, num_classes):
    def client_fn(cid: str) -> client.VowpalWabbitClient:
        client_data, client_aux = data[int(cid)], aux[int(cid)]
        model = init_model(num_features, num_classes)
        return client.VowpalWabbitClient(cid,
                                         model,
                                         opt_global_model,
                                         opt_local_models[int(cid)],
                                         client_data,
                                         client_aux)
    return client_fn

def create_evaluate_metrics_aggregation_fn(weights, client_id=0):
    def weighted_average_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        opt_reward = [weights[i] * metric["opt_reward"] for i, (_, metric) in enumerate(metrics)]
        reward = [weights[i] * metric["reward"] for i, (_, metric) in enumerate(metrics)]
        global_regret = [weights[i] * metric["global_regret"] for i, (_, metric) in enumerate(metrics)]
        local_regrets = {}
        for client_id in range(len(metrics)):
            local_regrets[client_id] = metrics[client_id][1]["local_regret"]
        return {"opt_reward": sum(opt_reward),
                "reward": sum(reward),
                "global_regret": sum(global_regret),
                "local_regrets": local_regrets}
    return weighted_average_aggregation_fn

def run_federated_learning(data, aux, opt_global_model, opt_local_models,
                           num_features, num_classes,
                           num_rounds=5, num_clients=5, client_weights=None):
    # Create client function
    client_fn = create_client_fn(data, aux,
                                 opt_global_model, opt_local_models,
                                 num_features, num_classes)

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
