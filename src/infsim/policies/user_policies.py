from typing import Callable

import numpy as np
import pandas as pd
from numpy.random import Generator

from infsim.utils.conversion_sampling import softmax_max_conversion_sampler


def user_level_control_policy(
    user_context: pd.DataFrame, item_context: pd.DataFrame, rng: Generator
) -> np.array:
    user_context = user_context.copy()
    item_context = item_context.drop(columns=["is_treated"], errors="ignore")
    user_treatments = user_context[["user_id"]]
    user_treatments["is_treated"] = rng.binomial(1, 0.5, len(user_treatments))
    item_context = pd.merge(item_context, user_treatments, on="user_id", how="left")
    item_context["model_score"] = 0.5

    return user_context, item_context


def ground_truth_based_user_level_uplift_policy(
    user_context: np.array,
    item_context: np.array,
    discount_percentage: float,
    rng: Generator,
    attractiveness_func: Callable,
    noise: float,
) -> np.array:
    """
    This method is an optimal user level uplift policy based on the (hidden) attractiveness exposed by the simulation.
    """
    user_context = user_context.copy()
    item_context = item_context.copy()
    item_context = item_context.drop(
        columns=["is_treated", "model_score"], errors="ignore"
    )
    item_context_ones = item_context.copy()
    item_context_ones["is_treated"] = 1
    item_context_zeros = item_context.copy()
    item_context_zeros["is_treated"] = 0

    item_context_ones = attractiveness_func(
        user_context, item_context_ones, discount_percentage=discount_percentage
    )
    item_context_zeros = attractiveness_func(
        user_context, item_context_zeros, discount_percentage=discount_percentage
    )

    temperature = 1  # Just a dummy value as this is not relevant for user level policy optimisation.
    user_context_ones, _ = softmax_max_conversion_sampler(
        user_context, item_context_ones, temperature=temperature
    )
    user_context_zeros, _ = softmax_max_conversion_sampler(
        user_context, item_context_zeros, temperature=temperature
    )

    conversion_treatment_effect = (
        user_context_ones._conversion_probability
        / user_context_zeros._conversion_probability
    )
    conversion_treatment_effect += rng.normal(
        scale=noise, size=conversion_treatment_effect.shape
    )

    _user_treatments = user_context[["user_id"]]
    _user_treatments["is_treated"] = np.array(
        conversion_treatment_effect > conversion_treatment_effect.mean()
    ).astype(int)
    _user_treatments["model_score"] = conversion_treatment_effect

    item_context = pd.merge(item_context, _user_treatments, on="user_id", how="left")

    return user_context, item_context
