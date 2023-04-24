import numpy as np

from sklearn.datasets import make_classification

import utils

def generate_csc_data(n_samples=1000,
                      n_clients=4,
                      n_classes=3,
                      n_features=10,
                      n_informative=5,
                      random_state=42):
    # Generate random contexts and labels
    X, y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=random_state)

    # Generate random costs
    costs = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y):
        for j in range(n_classes):
            if j != label:
                costs[i, j] = np.random.uniform(1, 5)

    # Convert data to VW format
    vw_data = utils.to_vw_format(X, y, costs)

    # Split data
    data = {client_id:client_data for client_id, client_data in enumerate(np.array_split(vw_data, n_clients))}
    costs = {client_id:client_costs for client_id, client_costs in enumerate(np.array_split(costs, n_clients))}

    return data, costs


def generate_observational_data(num_clients, sample_sizes, num_features, num_actions):
    np.random.seed(42)

    # num_clients = 4
    # num_actions = 4
    # num_features = 100
    # sample_sizes = [
    #     100,
    #     10000,
    #     1000,
    #     1000
    # ]
    
    context_means = 0.25 * np.array([
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1]
    ])
    cov = np.array([[1, 0], [0, 1]])

    reward_weights = 0.5 * np.array([
        [-1, 1],
        [1, -1],
        [-1, -1],
        [1, 1]
    ])

    reward_weights = np.random.multivariate_normal(mean=np.zeros(num_features), cov=np.eye(num_features), size=num_actions)

    action_params = np.random.randn(num_features, num_actions)
    action_params /= np.linalg.norm(action_params, axis=0)
    action_params = 0.5 * action_params

    data = {}
    aux = {}
    for client_id in range(num_clients):
        # Generate data
        num_samples = sample_sizes[client_id]
        # contexts = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=num_samples)
        contexts = np.random.uniform(low=-1, high=1, size=(num_samples, num_features))
        rewards_vectors = np.zeros((num_samples, num_actions))
        for i in range(num_samples):
            for a in range(num_actions):
                rewards_vectors[i, a] = np.dot(contexts[i], action_params[:, a])
                # rewards_vectors[i, a] = 1 / np.linalg.norm(contexts[i] - reward_weights[a])
                # rewards_vectors[i, a] = np.exp(1-1/np.linalg.norm(contexts[i]-reward_weights[a]))

        # actions = np.random.choice(num_actions, p=[0.7, 0.1, 0.1, 0.1], size=num_samples)
        actions = np.random.choice(num_actions, size=num_samples)
        epsilons = np.random.normal(scale=1, size=(num_samples, num_actions))
        rewards_vectors = rewards_vectors + epsilons
        rewards = rewards_vectors[np.arange(num_samples), actions]


        # Compute AIPW scores
        crossfit_map, mu, e = utils.cross_fit_nuisance_params(contexts, actions, rewards, num_actions)
        AIPW_vectors = utils.compute_AIPW_scores(contexts, actions, rewards, num_actions, crossfit_map, mu, e)

        # Convert data to VW format
        vw_data = utils.to_vw_format(contexts, actions, -AIPW_vectors)

        data[client_id] = vw_data
        # costs[client_id] = -AIPW_vector
        aux[client_id] = {"X": contexts, "A": actions, "Y": rewards, "costs": -rewards_vectors}
    
    return data, aux
