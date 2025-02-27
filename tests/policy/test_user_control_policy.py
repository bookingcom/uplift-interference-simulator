from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from infsim.environments.base_simulator import BaseEnvironment
from infsim.policies.user_policies import (
    user_level_control_policy,
    ground_truth_based_user_level_uplift_policy,
)
from infsim.policies.item_policies import (
    compute_item_treatment_attr_effect,
    ground_truth_based_item_level_uplift_policy,
    item_level_control_policy,
)
from infsim.utils.attractiveness import user_item_attractiveness
from infsim.utils.context_sampler import UniformContextSampler
from infsim.utils.conversion_sampling import softmax_max_conversion_sampler


def test_policy_creates_treatment_for_each_user_at_50_50_split():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(42)

    _, actual_policy_item_context = user_level_control_policy(
        user_context, item_context, rng
    )

    nr_treated_items = (
        actual_policy_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )

    # Validate that treatment is always given to all or no items for a given user
    assert np.all(np.logical_or(nr_treated_items == 0, nr_treated_items == _n_items))
    # Check if treatment is given approx 50% of the time.
    assert np.round(actual_policy_item_context["is_treated"].mean(), decimals=1) == 0.5


def test_policy_is_consistent_under_same_random_seed():
    seed = 42
    _n_samples = 10
    _n_items = 15
    _n_user_features = 11  # Ignored in this test case
    _n_item_features = 12  # Ignored in this test case

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    _, rng_1_treatment = user_level_control_policy(user_context, item_context, rng1)
    _, rng_2_treatment = user_level_control_policy(user_context, item_context, rng2)

    assert_frame_equal(rng_1_treatment, rng_2_treatment)


def test_policy_is_different_under_different_random_seed():
    _n_samples = 10
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(37)

    _, rng_1_treatment = user_level_control_policy(user_context, item_context, rng1)
    _, rng_2_treatment = user_level_control_policy(user_context, item_context, rng2)

    assert not np.equal(
        rng_1_treatment.is_treated.to_numpy(), rng_2_treatment.is_treated.to_numpy()
    ).all()


def test_user_level_ground_truth_policy_treats_all_or_no_items():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng = np.random.default_rng(37)
    attractiveness_func = partial(user_item_attractiveness, seed=37)
    _, treated_item_context = ground_truth_based_user_level_uplift_policy(
        user_context,
        item_context,
        _discount_percentage,
        rng,
        attractiveness_func,
        noise=0.02,
    )

    assert len(treated_item_context) == (_n_samples * _n_items)
    nr_treated_items = (
        treated_item_context.groupby("user_id").sum()["is_treated"].to_numpy()
    )
    assert np.all(np.logical_or(nr_treated_items == 0, nr_treated_items == _n_items))


def test_user_level_ground_truth_policy_is_identical_under_same_seed():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    _, treated_item_context_1 = ground_truth_based_user_level_uplift_policy(
        user_context,
        item_context,
        _discount_percentage,
        rng1,
        attractiveness_func,
        noise=0.02,
    )
    _, treated_item_context_2 = ground_truth_based_user_level_uplift_policy(
        user_context,
        item_context,
        _discount_percentage,
        rng2,
        attractiveness_func,
        noise=0.02,
    )

    assert_frame_equal(treated_item_context_1, treated_item_context_2)


def test_user_level_ground_truth_policy_is_different_under_different_seed():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(37)

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    _, treated_item_context_1 = ground_truth_based_user_level_uplift_policy(
        user_context,
        item_context,
        _discount_percentage,
        rng1,
        attractiveness_func,
        noise=0.02,
    )
    _, treated_item_context_2 = ground_truth_based_user_level_uplift_policy(
        user_context,
        item_context,
        _discount_percentage,
        rng2,
        attractiveness_func,
        noise=0.02,
    )

    assert np.not_equal(
        treated_item_context_1.is_treated, treated_item_context_2.is_treated
    ).any()


def test_user_level_ground_truth_policy_can_be_applied_on_existing_logs():
    context_sampler = UniformContextSampler(
        n_items=10, n_item_features=11, n_user_features=12, seed=43
    )

    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=user_item_attractiveness,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=0.1),
        policy=user_level_control_policy,
        seed=42,
    )

    # Generate 1000 training steps
    logs = env.step(1000)

    _, user_level_policy_item_context = ground_truth_based_user_level_uplift_policy(
        logs.user_context,
        logs.item_context,
        discount_percentage=logs.discount_percentage,
        rng=np.random.default_rng(42),
        attractiveness_func=partial(env.attractiveness_function, seed=env.seed),
        noise=0.00,
    )

    assert "is_treated" in user_level_policy_item_context.columns
    assert "model_score" in user_level_policy_item_context.columns


def test_compute_item_treatment_attractiveness_increase_provides_an_uplift_score_for_each_item():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    item_treatment_attr_effect = compute_item_treatment_attr_effect(
        attractiveness_func,
        user_context,
        item_context,
        0.08,
    )

    assert len(item_treatment_attr_effect) == (_n_samples * _n_items)
    assert (item_treatment_attr_effect < 1.0).sum() > 0
    assert (item_treatment_attr_effect > 1.0).sum() > 0


def test_item_level_ground_truth_policy_descreases_profit_using_noise_parameter():
    profit = []
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    context_sampler_1 = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )

    attractiveness_func = partial(user_item_attractiveness, seed=42)

    env = BaseEnvironment(
        context_sampler=context_sampler_1,
        attractiveness_function=attractiveness_func,
        policy=item_level_control_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
    )
    logs = env.step(_n_samples)

    for i in range(20):
        policy_low_noise = partial(
            ground_truth_based_item_level_uplift_policy,
            discount_percentage=0.30,
            attractiveness_func=attractiveness_func,
            noise=0.0 + (0.01 * i),
        )

        resampled_logs = env.resample_with_alternative_policy(logs, policy_low_noise)
        profit.append(resampled_logs.expected_profit.sum())

    # Profit is not always higher with less noise, but the trend is always correct
    assert np.sum(profit[:10]) > np.sum(profit[10:])


def test_user_level_control_policy_creates_treatment_for_each_user_at_50_50_split():
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1},
            {"user_id": 1, "item_id": 2},
            {"user_id": 1, "item_id": 3},
            {"user_id": 2, "item_id": 4},
            {"user_id": 2, "item_id": 5},
        ]
    )

    user_context = pd.DataFrame(
        [
            {"user_id": 1},
            {"user_id": 2},
        ]
    )

    rng = np.random.default_rng(42)

    _, actual_treatment = user_level_control_policy(user_context, item_context, rng)

    # Validate that treatment is always given to all or no items for a given user
    assert set(actual_treatment.groupby("user_id")["is_treated"].mean()) == {0, 1}
    # Check if treatment is given approx 50% of the time.
    assert (actual_treatment["model_score"] == 0.5).all()


def test_user_level_control_policy_is_consistent_under_same_random_seed():
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1},
            {"user_id": 1, "item_id": 2},
            {"user_id": 1, "item_id": 3},
            {"user_id": 2, "item_id": 4},
            {"user_id": 2, "item_id": 5},
        ]
    )

    user_context = pd.DataFrame(
        [
            {"user_id": 1},
            {"user_id": 2},
        ]
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    _, rng_1_treatment = user_level_control_policy(user_context, item_context, rng1)
    _, rng_2_treatment = user_level_control_policy(user_context, item_context, rng2)

    assert_frame_equal(rng_1_treatment, rng_2_treatment)


def test_user_level_control_policy_is_different_under_different_random_seed():
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1},
            {"user_id": 1, "item_id": 2},
            {"user_id": 1, "item_id": 3},
            {"user_id": 2, "item_id": 4},
            {"user_id": 2, "item_id": 5},
        ]
    )

    user_context = pd.DataFrame(
        [
            {"user_id": 1},
            {"user_id": 2},
        ]
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(37)

    rng_1_treatment = user_level_control_policy(user_context, item_context, rng1)
    rng_2_treatment = user_level_control_policy(user_context, item_context, rng2)

    with pytest.raises(AssertionError):
        assert_frame_equal(rng_1_treatment, rng_2_treatment)
