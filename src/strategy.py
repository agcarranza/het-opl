from logging import WARNING

import flwr as fl

from flwr.common import (
    FitRes,
    Scalar,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union


class WeightedFedAvg(fl.server.strategy.FedAvg):
    """Custom Weighted FedAvg strategy implementation."""
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Calculate the weighted average of the parameters
        parameters_results = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        weighted_parameters = [
            [weight * param for param in params]
            for params, weight in zip(parameters_results, self.weights)
        ]
        aggregated_params = [
            sum(param_list) for param_list in zip(*weighted_parameters)
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregated_params)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
