import numpy as np
import pytest

from infsim.policies.item_policies import (
    item_level_control_policy,
    ground_truth_based_item_level_uplift_policy,
)
from infsim.utils.attractiveness import user_item_attractiveness
from infsim.utils.context_sampler import UniformContextSampler


def get_context_sampler(n_items=10, n_user_features=11, n_item_features=12):
    context_sampler = UniformContextSampler(
        n_items=n_items,
        n_user_features=n_user_features,
        n_item_features=n_item_features,
        seed=42,
    )

    return context_sampler


def test_item_policy_with_100_percent_fraction_is_identical_to_user_level_treating_all():
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    _, actual_policy_item_context = item_level_control_policy(
        user_context, item_context, rng, treatment_fraction=1.0
    )

    nr_treated_items = (
        actual_policy_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )

    # Validate that treatment is always given to all items for a given user
    assert np.all(nr_treated_items == _n_items)


@pytest.mark.parametrize("treatment_fraction", [0.1, 0.5, 1.0])
def test_item_policy_with_specified_treatment_fraction_treated(treatment_fraction):
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    _, actual_policy_item_context = item_level_control_policy(
        user_context, item_context, rng, treatment_fraction=treatment_fraction
    )

    # Validate that treatment is always given to all items for a given user
    assert (
        np.round(actual_policy_item_context["is_treated"].mean(), decimals=1)
        == treatment_fraction
    )


@pytest.mark.parametrize("treatment_fraction", [0.1, 0.5, 1.0])
def test_ground_truth_item_policy_with_provided_fraction_treated(treatment_fraction):
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    _, actual_policy_item_context = ground_truth_based_item_level_uplift_policy(
        user_context,
        item_context,
        rng,
        attractiveness_func=user_item_attractiveness,
        discount_percentage=0.08,
        treatment_fraction=treatment_fraction,
        noise=0.0,
    )

    # Validate that treatment is always given to all items for a given user
    assert (
        np.round(actual_policy_item_context["is_treated"].mean(), decimals=1)
        == treatment_fraction
    )
