from typing import Callable, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator

from infsim.environments.base_simulator import (
    get_price_from_uniform_random_context,
    get_commission_percentage_from_uniform_random_context,
)


def item_level_control_policy(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    rng: Generator,
    treatment_fraction: float = 0.5,
) -> np.array:
    user_context = user_context.copy()
    item_context = item_context.drop(columns=["is_treated"], errors="ignore")

    item_context["is_treated"] = rng.binomial(1, treatment_fraction, len(item_context))
    item_context["model_score"] = treatment_fraction

    return user_context, item_context


def compute_item_treatment_attr_effect(
    attractiveness_func: Callable,
    user_context: np.array,
    item_context: np.array,
    discount_percentage: float,
) -> np.array:
    """
    It would be too computationally expensive to compute the attractiveness increase of all permutations of treatments.
    Instead, we do a best-effort estimation of treatment value.

    We compute the treatment of each item, post-treatment, and compare it to the highest pre-treatment attractiveness
     for each user. Items that are higher post-treatment than the highest pre-treatment attractiveness will have a
     positive value, while the items with lower post-treatment attractiveness will have a negative value.
    """
    item_context = item_context.copy()
    item_context["is_treated"] = 1
    item_context = attractiveness_func(user_context, item_context, discount_percentage)

    max_base_attractiveness_treatment_effect = (
        item_context.groupby("user_id")
        .max()[["_base_attractiveness"]]
        .rename(columns={"_base_attractiveness": "_max_base_attractiveness"})
    )
    item_context = pd.merge(
        item_context, max_base_attractiveness_treatment_effect, on="user_id", how="left"
    )

    item_attractiveness_treatment_effect = (
        item_context._treatment_attractiveness / item_context._max_base_attractiveness
    )

    return item_attractiveness_treatment_effect


def ground_truth_based_item_level_uplift_policy(
    user_context: np.array,
    item_context: np.array,
    rng: Generator,
    discount_percentage: float,
    attractiveness_func: Callable,
    noise: float,
    treatment_fraction: Optional[float] = None,
):
    """
    This is the most simply closed form ground-truth item level treatment strategy I could come up with. However, it
    is not perfect. It attempts to measure the effect on each item treatment on the base conversion and multiply it with the
    profit of the treated item. Any item that then increases overall profit is treated.
    """
    item_context = item_context.copy()
    user_context = user_context.copy()
    treatment_attractiveness_effect = compute_item_treatment_attr_effect(
        attractiveness_func,
        user_context,
        item_context,
        discount_percentage=discount_percentage,
    )

    price = get_price_from_uniform_random_context(item_context)
    commission = get_commission_percentage_from_uniform_random_context(item_context)

    base_profit = price * commission

    post_treatment_profit = base_profit * treatment_attractiveness_effect

    uplift_scores = post_treatment_profit / base_profit

    uplift_scores_noise = rng.uniform(
        low=uplift_scores.min(), high=uplift_scores.max(), size=uplift_scores.shape
    )
    noisy_uplift_score = ((1 - noise) * uplift_scores) + (noise * uplift_scores_noise)

    if treatment_fraction is not None:
        # Treat the top fraction_to_treat items according to the rank of the uplift scores
        num_to_treat = int(treatment_fraction * len(item_context))
        # Get threshold for uplift scores
        threshold = np.sort(noisy_uplift_score)[::-1][num_to_treat - 1]
        item_context["is_treated"] = noisy_uplift_score > threshold
    else:
        item_context["is_treated"] = noisy_uplift_score > 1.0
    item_context["model_score"] = noisy_uplift_score

    return user_context, item_context
