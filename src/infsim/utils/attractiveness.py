import numpy as np
import pandas as pd


def base_user_item_attractiveness(
    user_context: np.array, item_context: np.array, seed: int = None
) -> np.array:
    """
    Basic linear user-item attractiveness score. Weights are seeded and are used to compute an interaction between
    item and user context. Attractiveness is rescaled between 0 and 1 using on the number of features.


    :param user_context: n_samples X n_user_features, should be sampled at uniform random
    :param item_context: n_samples X n_items x n_item_features, should be sampled at uniform random
    :param seed: integer to ensure consistent weights are created between calls
    :return: n_samples x n_items with attractiveness for each user-item pair
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(size=(user_context.shape[1], item_context.shape[2]))
    attractiveness = np.einsum(
        "ij,ikj->ik", np.dot(user_context, weights), item_context
    )

    return attractiveness / (item_context.shape[2] * user_context.shape[1])


def user_item_attractiveness(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    discount_percentage: float,
    seed: int = None,
) -> pd.DataFrame:
    item_context = item_context.copy()

    # Convert the pandas dataframe to numpy for efficiency.
    user_features = [col for col in user_context.columns if "user_feat" in col]
    item_features = [col for col in item_context.columns if "item_feat" in col]
    user_context_numpy = user_context[user_features].to_numpy()
    item_context_numpy = np.stack(
        item_context.groupby("user_id")[item_features]
        .apply(lambda x: x.to_numpy())
        .to_numpy()
    )

    base_attractiveness = base_user_item_attractiveness(
        user_context_numpy, item_context_numpy, seed=seed
    )
    if seed:
        delta_seed = seed + 1
    else:
        delta_seed = None
    treatment_attractiveness_delta = base_user_item_attractiveness(
        user_context_numpy, item_context_numpy, seed=delta_seed
    )

    treatment_attractiveness_delta = np.clip(
        treatment_attractiveness_delta,
        a_min=0,
        a_max=0.5,
    )

    # Set 50 percent of elements in treatment_attractiveness_delta to zero
    treatment_attractiveness_delta = treatment_attractiveness_delta * (
        (treatment_attractiveness_delta * 10**10).astype(int) % 2
    )

    item_context["_base_attractiveness"] = base_attractiveness.ravel()
    item_context["_treatment_attractiveness_delta"] = (
        treatment_attractiveness_delta.ravel()
    )
    item_context["_treatment_attractiveness"] = item_context["_base_attractiveness"] + (
        item_context["is_treated"]
        * discount_percentage
        * 10
        * item_context["_treatment_attractiveness_delta"]
    )

    return item_context


def attractiveness_from_pandas(
    _: pd.DataFrame, item_context: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    item_context["_treatment_attractiveness"] = (
        item_context["_base_attractiveness"]
        + item_context["is_treated"] * item_context["_treatment_attractiveness_delta"]
    )

    return item_context
