import re
from functools import partial

import numpy as np
import pytest

from infsim.environments.base_simulator import BaseEnvironment, PolicyLogs
from infsim.policies.fractional_policies import (
    fractional_control_policy,
    ground_truth_fractional_policy,
)
from infsim.policies.user_policies import user_level_control_policy
from infsim.utils.attractiveness import user_item_attractiveness
from infsim.utils.context_sampler import UniformContextSampler
from infsim.utils.conversion_sampling import softmax_max_conversion_sampler


def get_context_sampler(n_items=10, n_user_features=11, n_item_features=12):
    context_sampler = UniformContextSampler(
        n_items=n_items,
        n_user_features=n_user_features,
        n_item_features=n_item_features,
        seed=42,
    )

    return context_sampler


def test_fractional_policy_with_100_percent_fraction_is_identical_to_user_level_control():
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    _, actual_policy_item_context = fractional_control_policy(
        user_context, item_context, rng, treatment_fraction=1.0
    )

    nr_treated_items = (
        actual_policy_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )

    # Validate that treatment is always given to all or no items for a given user
    assert np.all(np.logical_or(nr_treated_items == 0, nr_treated_items == _n_items))
    # Check if treatment is given approx 50% of the time.
    assert np.round(actual_policy_item_context["is_treated"].mean(), decimals=1) == 0.5


@pytest.mark.parametrize("treatment_fraction", [0.1, 0.5, 0.9])
def test_fractional_policy_treats_percentage_as_specified_for_50_percent_of_users(
    treatment_fraction,
):
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    user_context, actual_policy_item_context = fractional_control_policy(
        user_context, item_context, rng, treatment_fraction=treatment_fraction
    )

    nr_treated_items = (
        actual_policy_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )

    # Validate that treatment is always given to all or no items for a given user
    assert np.all(nr_treated_items == user_context["treatment_fraction"] * _n_items)
    # Check if treatment is given approx 50% of the time.
    assert (
        np.round(
            (user_context["treatment_fraction"] == treatment_fraction).mean(),
            decimals=1,
        )
        == 0.5
    )


def test_fractional_policy_raises_when_fraction_is_higher_than_1():
    _n_samples = 10
    context_sampler = get_context_sampler()
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    with pytest.raises(
        RuntimeError, match=re.escape("treatment_fraction cannot be larger than 1.")
    ):
        _ = fractional_control_policy(
            user_context, item_context, rng, treatment_fraction=1.1
        )


def test_fractional_policy_raises_when_fraction_is_lower_or_equal_to_0():
    _n_samples = 10
    context_sampler = get_context_sampler()
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    with pytest.raises(
        RuntimeError,
        match=re.escape("treatment_fraction cannot be smaller then or equal to 0."),
    ):
        _ = fractional_control_policy(
            user_context, item_context, rng, treatment_fraction=0
        )


def test_fractional_policy_raises_when_perfect_fraction_cannot_be_assigned():
    _n_samples = 10
    _n_items = 9
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    with pytest.raises(
        RuntimeError,
        match=re.escape("Non-exact treatment fractions are not supported yet."),
    ):
        _ = fractional_control_policy(
            user_context, item_context, rng, treatment_fraction=0.1
        )


@pytest.mark.parametrize("treatment_fraction", [0.1, 0.5, 0.9])
def test_ground_truth_fractional_policy_treats_only_the_given_fraction(
    treatment_fraction,
):
    _n_samples = 1000
    _n_items = 10
    context_sampler = get_context_sampler(n_items=_n_items)
    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    user_context, actual_policy_item_context = ground_truth_fractional_policy(
        user_context,
        item_context,
        rng,
        attractiveness_func=attractiveness_func,
        treatment_fraction=treatment_fraction,
        discount_percentage=0.08,
    )

    nr_treated_items = (
        actual_policy_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )

    # Validate that treatment is consistent with given treatment fraction
    assert np.all(nr_treated_items == (user_context["treatment_fraction"] * _n_items))


def test_ground_truth_fractional_policy_gives_same_treatments_for_two_identical_contexts():
    treatment_fraction = 1 / 3
    _n_samples = 1
    _n_items = 9
    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_item_features=2,
        n_user_features=2,
        seed=42,  # 42
    )

    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=partial(user_item_attractiveness, seed=42),
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=0.1),
        policy=user_level_control_policy,  # control_policy is not used for evaluation
        seed=43,  # 43,
    )

    full_logs = env.step(2)

    user_id_b = full_logs.user_context["user_id"][1]

    part_b = PolicyLogs(
        item_context=full_logs.item_context[
            full_logs.item_context["user_id"] == user_id_b
        ]
        .reset_index(drop=True)
        .copy(),
        user_context=full_logs.user_context[
            full_logs.user_context["user_id"] == user_id_b
        ]
        .reset_index(drop=True)
        .copy(),
        discount_percentage=full_logs.discount_percentage,
    )

    part_c = PolicyLogs(
        item_context=full_logs.item_context[
            full_logs.item_context["user_id"] == user_id_b
        ]
        .reset_index(drop=True)
        .copy(),
        user_context=full_logs.user_context[
            full_logs.user_context["user_id"] == user_id_b
        ]
        .reset_index(drop=True)
        .copy(),
        discount_percentage=full_logs.discount_percentage,
    )
    _, item_context_1 = ground_truth_fractional_policy(
        part_b.user_context.reset_index(drop=True),
        part_b.item_context.reset_index(drop=True),
        np.random.default_rng(42),
        attractiveness_func=partial(user_item_attractiveness, seed=42),
        treatment_fraction=treatment_fraction,
        discount_percentage=0.08,
    )

    _, item_context_2 = ground_truth_fractional_policy(
        part_c.user_context.reset_index(drop=True),
        part_c.item_context.reset_index(drop=True),
        np.random.default_rng(42),
        attractiveness_func=partial(user_item_attractiveness, seed=42),
        treatment_fraction=treatment_fraction,
        discount_percentage=0.08,
    )

    assert (item_context_2["model_score"] == item_context_1["model_score"]).all()


def test_ground_truth_fractional_policy_treats_items_with_more_profit():
    _n_samples = 1000
    _n_items = 10
    treatment_fraction = 0.5

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    fractional_policy = partial(
        ground_truth_fractional_policy,
        treatment_fraction=treatment_fraction,
        discount_percentage=0.08,
        attractiveness_func=attractiveness_func,
    )

    env = BaseEnvironment(
        context_sampler=get_context_sampler(n_items=_n_items),
        attractiveness_function=attractiveness_func,
        policy=fractional_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
    )

    policy_logs = env.step(_n_samples)

    inverse_policy_logs = env.resample_with_alternative_policy(
        policy_logs, treatment=1 - policy_logs.item_context.is_treated
    )

    assert policy_logs.expected_profit.sum() > inverse_policy_logs.expected_profit.sum()


@pytest.mark.skip("noise is not yet implemented yet.")
def test_ground_truth_fractional_policy_reduces_profit_when_noise_is_increased():
    pass
