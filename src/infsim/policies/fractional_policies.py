from typing import Callable

import numpy as np
import pandas as pd
from numpy.random import Generator

from infsim.policies.item_policies import ground_truth_based_item_level_uplift_policy


def fractional_control_policy(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    rng: Generator,
    treatment_fraction: float,
) -> np.array:
    check_treatment_fraction_viability(item_context, treatment_fraction)

    user_context = user_context.copy()
    item_context = item_context.drop(columns=["is_treated"], errors="ignore")
    user_treatments = user_context[["user_id"]]
    user_treatments["_is_treated_user"] = rng.binomial(1, 0.5, len(user_treatments))
    user_context["treatment_fraction"] = (
        user_treatments["_is_treated_user"] * treatment_fraction
    )

    def assign_random_numbers(group):
        group["_random_number"] = np.random.permutation(len(group)) + 1
        group["_n_items"] = len(group)
        return group

    # Provide each item with a random number between 1 and N
    _item_treatment_assigment = (
        item_context.groupby("user_id")
        .apply(assign_random_numbers)
        .reset_index(drop=True)
    )

    # Treat only the items with a random number up until the to-be-treated fraction
    item_context["_is_treated_item"] = (
        _item_treatment_assigment["_random_number"]
        <= _item_treatment_assigment["_n_items"] * treatment_fraction
    )

    item_context = pd.merge(item_context, user_treatments, on="user_id", how="left")
    item_context["is_treated"] = (
        item_context["_is_treated_user"] & item_context["_is_treated_item"]
    )
    item_context = item_context.drop(columns=["_is_treated_item", "_is_treated_user"])

    item_context["model_score"] = 0.5

    return user_context, item_context


def ground_truth_fractional_policy(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    rng: Generator,
    treatment_fraction: float,
    discount_percentage: float,
    attractiveness_func: Callable,
) -> np.array:
    """
    This method allows to construct a ground truth fractional policy that treats a given fraction of items for each
    user.
    """
    user_context = user_context.copy()
    item_context = item_context.copy()

    check_treatment_fraction_viability(item_context, treatment_fraction)

    user_context, item_context = ground_truth_based_item_level_uplift_policy(
        user_context,
        item_context,
        rng,
        discount_percentage,
        attractiveness_func,
        noise=0.0,
    )

    item_context = item_context.reset_index().drop(columns=["is_treated"])

    def get_item_rank_in_group(group):
        group["_score_rank"] = group["model_score"].rank(
            method="dense", ascending=False
        )
        group["_n_items"] = len(group)
        return group

    # item_context = item_context.reset_index()
    _item_treatment_assigment = (
        item_context.groupby("user_id")
        .apply(get_item_rank_in_group)
        .reset_index(drop=True)
    )

    _item_treatment_assigment["is_treated"] = (
        _item_treatment_assigment["_score_rank"]
        <= _item_treatment_assigment["_n_items"] * treatment_fraction
    )

    item_context = pd.merge(
        item_context,
        _item_treatment_assigment[["user_id", "index", "is_treated"]],
        on=["user_id", "index"],
        how="left",
    )
    item_context = item_context.drop(columns=["index"])
    user_context["treatment_fraction"] = treatment_fraction

    return user_context, item_context


def check_treatment_fraction_viability(
    item_context: pd.DataFrame, treatment_fraction: float
) -> None:
    if treatment_fraction > 1.0:
        raise RuntimeError("treatment_fraction cannot be larger than 1.")
    if treatment_fraction <= 0.0:
        raise RuntimeError("treatment_fraction cannot be smaller then or equal to 0.")
    if (
        item_context.groupby("user_id")["user_id"].count() * treatment_fraction % 1 != 0
    ).any():
        raise NotImplementedError(
            "Non-exact treatment fractions are not supported yet."
        )
