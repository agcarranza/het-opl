import numpy as np

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression


def to_vw_format(X, y=None, costs=None, idx_to_weight_mapping=None):
    """Convert context, labels, and costs data to cost-sensitive VW format."""
    data = []
    for i in range(len(X)):
        features = ' '.join([f"{j}:{X[i, j]}" for j in range(len(X[i]))])
        if y is None:
            data.append(f"| {features}")
        else:
            labels = ' '.join([f"{j + 1}:{costs[i, j]}" for j in range(len(costs[i]))])
            if idx_to_weight_mapping is None:
                data.append(f"{labels} 1 | {features}")
            else:
                weight = idx_to_weight_mapping[i]
                data.append(f"{labels} {weight} | {features}")
    return data

def cross_fit_nuisance_params(X, A, Y, num_actions, num_folds=5):
    """Estimate nuisance parameters with cross-fitting."""
    mean_response_models, propensity_models = [], []
    fold_to_indices_mapping = {}

    kf = KFold(n_splits=num_folds)
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        # Train on train_index
        X_train = X[train_index]
        A_train = A[train_index]
        Y_train = Y[train_index]

        # Add mapping of fold to test indices
        fold_to_indices_mapping[fold] = test_index

        # Estimate the mean response model: E[Y|X,A]
        action_specific_mean_response_models = []
        for a in range(num_actions):
            mask = A_train == a
            action_specific_mean_response_model = LinearRegression().fit(X_train[mask],
                                                                         Y_train[mask])
            action_specific_mean_response_models.append(action_specific_mean_response_model)
        mean_response_models.append(action_specific_mean_response_models)
        
        # Estimate the propensity model: P[A|X]
        propensity_model = LogisticRegression(multi_class="multinomial",
                                              solver="lbfgs",
                                              max_iter=1000).fit(X_train, np.ravel(A_train))
        propensity_models.append(propensity_model)

    return fold_to_indices_mapping, mean_response_models, propensity_models

def compute_AIPW_scores(X, A, Y,
                        num_actions,
                        fold_to_indices_mapping,
                        mean_response_models,
                        propensity_models,
                        eta=1e-8):
    """Compute AIPW scores given nuisance parameter estimates."""
    num_samples = len(X)
    AIPW_scores = np.zeros((num_samples, num_actions))
    mean_response = np.zeros((num_samples, num_actions))
    for k, mask in fold_to_indices_mapping.items():
        # Compute mean response
        for a in range(num_actions):
            mean_response[mask, a] += mean_response_models[k][a].predict(X[mask])

        # Compute inverse propensity weights (with clipping)
        propensities = propensity_models[k].predict_proba(X[mask])
        if eta:
            propensities = np.clip(propensities[np.arange(len(propensities)), A[mask]],
                                   a_min=eta,
                                   a_max=None)
        inv_propensities = np.reciprocal(propensities)

        # Compute AIPW scores
        centered_rewards = Y[mask] - mean_response[mask, A[mask]]
        AIPW_scores[mask, A[mask]] += np.multiply(centered_rewards, inv_propensities)
        AIPW_scores[mask] += mean_response[mask]

    return AIPW_scores

def compute_regret(X, model, opt_costs, opt_model=None, idx_to_weight_mapping=None):
    """Compute regret metrics against (true or model) optimal outcome."""
    y_pred = model.predict(X)
    if opt_model:
        y_opt = opt_model.predict(X)

    opt_reward = 0
    reward = 0
    regret = 0
    n_samples = len(y_pred)
    for i in range(n_samples):
        if opt_model:
            opt_reward_i = -opt_costs[i, int(y_opt[i])-1]
        else:
            opt_reward_i = np.max(-opt_costs[i,:])
        reward_i = -opt_costs[i, int(y_pred[i])-1]
        if idx_to_weight_mapping is None:
            opt_reward += opt_reward_i
            reward += reward_i
            regret += opt_reward_i - reward_i
        else:
            weight = idx_to_weight_mapping[i]
            opt_reward += opt_reward_i * weight
            reward += reward_i * weight
            regret += (opt_reward_i - reward_i) * weight

    return regret / n_samples, opt_reward / n_samples, reward / n_samples