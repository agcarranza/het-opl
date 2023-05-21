import flwr as fl

from scipy.sparse import csr_matrix

import utils


class VowpalWabbitClient(fl.client.NumPyClient):
    def __init__(self, cid, model, opt_global_model, opt_local_model, data, aux):
        self.cid = cid
        self.model = model
        self.opt_global_model = opt_global_model
        self.opt_local_model = opt_local_model
        self.data = data
        self.aux = aux

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return [self.model.get_coefs().todense()]

    def set_parameters(self, parameters):
        self.model.set_coefs(csr_matrix(parameters[0]))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.data)
        print(f"Client {self.cid}, training finished for round {config['server_round']}")
        return self.get_parameters(config), len(self.data), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.set_parameters(parameters)

        X_test_vw = utils.to_vw_format(self.aux["X_test"])
        global_regret, opt_reward, reward = utils.compute_regret(X_test_vw,
                                                                 self.aux["true_costs_test"],
                                                                 self.model,
                                                                 self.opt_global_model)
        local_regret, _, _ = utils.compute_regret(X_test_vw,
                                                  self.aux["true_costs_test"],
                                                  self.model,
                                                  self.opt_local_model)

        return global_regret, len(X_test_vw), {"opt_reward": opt_reward,
                                               "reward": reward,
                                               "global_regret": global_regret,
                                               "local_regret": local_regret}
