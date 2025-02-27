import re
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from infsim.environments.base_simulator import BaseEnvironment, PolicyLogs
from infsim.policies.user_policies import (
    user_level_control_policy,
    ground_truth_based_user_level_uplift_policy,
)
from infsim.utils.attractiveness import (
    user_item_attractiveness,
    attractiveness_from_pandas,
)
from infsim.utils.context_sampler import UniformContextSampler, PandasContextIterator
from infsim.utils.conversion_sampling import softmax_max_conversion_sampler


def _get_dummy_environment():
    _n_items = 10
    _n_item_features = 11
    _n_user_features = 12

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_item_features=_n_item_features,
        n_user_features=_n_user_features,
        seed=43,
    )
    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=user_item_attractiveness,
        policy=user_level_control_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
        seed=42,
    )

    return env


@pytest.fixture
def dummy_environment():
    return _get_dummy_environment()


def holdout_treatment_policy(_, item_context):
    return np.zeros(shape=(item_context.shape[0], item_context.shape[1]))


def _sample_seeded_context(
    _n_items, _n_samples, n_item_features=11, n_user_features=12, seed=42
):
    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_item_features=n_item_features,
        n_user_features=n_user_features,
        seed=seed,
    )
    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=user_item_attractiveness,
        policy=user_level_control_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
    )
    dataset = env.step(_n_samples)
    return dataset


def test_policy_logs_price_is_between_20_and_120():
    _n_items = 10
    _n_samples = 1000

    dataset = _sample_seeded_context(_n_items, _n_samples)

    assert len(dataset.price) == (_n_samples * _n_items)
    assert np.logical_and(dataset.price > 20, dataset.price < 120).all()


def test_policy_logs_commission_is_between_10_and_15_percent():
    # We could potentially make this configurable in the future
    _n_items = 10
    _n_samples = 1000

    dataset = _sample_seeded_context(_n_items, _n_samples)

    assert len(dataset.commission_percentage) == (_n_samples * _n_items)
    assert np.logical_and(
        dataset.commission_percentage > 10, dataset.commission_percentage < 15
    ).all()


def test_policy_logs_expected_profit_is_higher_when_discount_is_reduced_given_same_attractiveness(
    dummy_environment,
):
    logs = dummy_environment.step(1000)

    regular_expected_profit = logs.expected_profit.sum()
    logs.discount_percentage = 0.05
    discount_expected_profit = logs.expected_profit.sum()

    assert regular_expected_profit < discount_expected_profit


def test_two_identical_environments_create_identical_logs():
    env1 = _get_dummy_environment()
    env2 = _get_dummy_environment()

    logs1 = env1.step(100)
    logs2 = env2.step(100)

    assert_frame_equal(logs1.user_context, logs2.user_context)
    assert_frame_equal(logs1.item_context, logs2.item_context)
    assert logs1.discount_percentage == logs2.discount_percentage


def test_resample_with_alternative_policy_returns_identical_attractiveness_with_identical_policy(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    dummy_environment.rng = np.random.default_rng(
        42
    )  # Reset the rng to get the same results
    logs_resample = dummy_environment.resample_with_alternative_policy(
        logs, dummy_environment.policy
    )

    assert_frame_equal(logs.user_context, logs_resample.user_context, check_like=True)
    assert_frame_equal(logs.item_context, logs_resample.item_context, check_like=True)


def test_resample_with_alternative_policy_raises_exception_when_both_policy_and_treatments_are_provided(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    with pytest.raises(RuntimeError, match="Both policy and treatment were provided"):
        _ = dummy_environment.resample_with_alternative_policy(
            logs, dummy_environment.policy, np.zeros_like(logs.item_context.is_treated)
        )


def test_resample_with_alternative_policy_raises_exception_when_neither_policy_and_treatments_are_provided(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    with pytest.raises(
        RuntimeError, match="Either policy or treatment should be provided"
    ):
        _ = dummy_environment.resample_with_alternative_policy(logs)


def test_resample_with_alternative_policy_raises_exception_when_treatment_is_of_different_size_than_original(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    with pytest.raises(
        RuntimeError,
        match=re.escape("Treatment is of incorrect size, expecting 1000, but got 100"),
    ):
        _ = dummy_environment.resample_with_alternative_policy(
            logs, treatment=np.zeros(shape=(100))
        )


def test_resample_with_alternative_policy_sets_model_scores_to_zero_if_treatment_is_provided(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    resampled_logs = dummy_environment.resample_with_alternative_policy(
        logs, treatment=np.zeros_like(logs.item_context.is_treated)
    )
    resampled_logs.model_scores = None


def test_resample_with_alternative_policy_returns_different_expected_profit_with_new_treatments(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    logs_resample = dummy_environment.resample_with_alternative_policy(
        logs, treatment=np.zeros_like(logs.item_context.is_treated)
    )

    assert logs.expected_profit.sum() != logs_resample.expected_profit.sum()


def test_resample_with_alternative_policy_returns_different_expected_profit_with_new_policy(
    dummy_environment,
):
    logs = dummy_environment.step(100)
    logs_resample = dummy_environment.resample_with_alternative_policy(
        logs, dummy_environment.policy
    )

    assert logs.expected_profit.sum() != logs_resample.expected_profit.sum()


def test_base_simulator_can_use_pandas_context_iterator_and_attractiveness_function_with_generic_policy_and_conv():
    user_context = pd.DataFrame(
        columns=["user_id", "_base_attractiveness", "_treatment_attractiveness_delta"],
        data=[(1, 0.2, 0.2), (2, 0.2, 0.2), (3, 0.2, 0.2), (4, 0.2, 0.2)],
    )

    item_context = pd.DataFrame(
        columns=["user_id", "_base_attractiveness", "_treatment_attractiveness_delta"],
        data=[(1, 0.2, 0.2), (2, 0.2, 0.2), (3, 0.2, 0.2), (4, 0.2, 0.2)],
    )

    context_iterator = PandasContextIterator(user_context, item_context)

    env = BaseEnvironment(
        context_sampler=context_iterator,
        attractiveness_function=attractiveness_from_pandas,
        policy=user_level_control_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
    )

    env.step(4)


def test_policy_logs_uses_price_from_context_if_column_exists():
    user_context = pd.DataFrame(columns=["user_id"], data=[(1,), (2,), (3,)])

    item_context = pd.DataFrame(
        columns=["user_id", "price"], data=[(1, 23), (2, 38), (3, 0.93)]
    )
    logs = PolicyLogs(user_context=user_context, item_context=item_context)

    assert (item_context.price == logs.price).all()


def test_policy_logs_uses_commission_percentage_from_context_if_column_exists():
    user_context = pd.DataFrame(columns=["user_id"], data=[(1,), (2,), (3,)])

    item_context = pd.DataFrame(
        columns=["user_id", "commission_percentage"], data=[(1, 23), (2, 38), (3, 0.93)]
    )
    logs = PolicyLogs(user_context=user_context, item_context=item_context)

    assert (item_context.commission_percentage == logs.commission_percentage).all()


def test_base_simulator_can_use_pandas_context_sampler_and_attractiveness_function():
    context_sampler = UniformContextSampler(
        n_items=10, n_item_features=11, n_user_features=12, seed=42
    )

    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=user_item_attractiveness,
        policy=user_level_control_policy,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=1),
    )

    logs = env.step(2000)

    assert logs.item_context.is_converted.sum() > 0
    assert round(logs.item_context.is_treated.mean(), ndigits=1) == 0.5


def test_base_environment_raises_if_seed_is_identical_to_context_sampler():
    # This is a really weird one, but when the seed of the context sampler and base environment are identical
    # the policy randomisation is not independent to the relative uplift, causing all kinds of issues.
    # I'm not sure why this is happening, but for now we prevent this behaviour with this hack.
    _n_samples = 100
    _n_items = 10
    context_sampler = UniformContextSampler(
        n_items=_n_items, n_item_features=11, n_user_features=12, seed=42
    )

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Context sampler and base environment cannot have the same seed"
        ),
    ):
        _ = BaseEnvironment(
            context_sampler=context_sampler,
            attractiveness_function=user_item_attractiveness,
            conversion_sampler=partial(softmax_max_conversion_sampler, temperature=0.1),
            policy=user_level_control_policy,
            seed=42,
        )


def test_ground_truth_policy_is_uncorrelated_with_treatments_from_random_policy():
    _n_samples = 1000
    _n_items = 10
    context_sampler = UniformContextSampler(
        n_items=_n_items, n_item_features=11, n_user_features=12, seed=42
    )

    env = BaseEnvironment(
        context_sampler=context_sampler,
        attractiveness_function=user_item_attractiveness,
        conversion_sampler=partial(softmax_max_conversion_sampler, temperature=0.1),
        policy=user_level_control_policy,
        seed=43,
    )

    # Generate 1000 training steps
    logs = env.step(_n_samples)

    _, user_level_policy_item_context = ground_truth_based_user_level_uplift_policy(
        logs.user_context,
        logs.item_context,
        discount_percentage=logs.discount_percentage,
        rng=env.rng,  # np.random.default_rng(42),
        attractiveness_func=partial(env.attractiveness_function, seed=env.seed),
        noise=0.00,
    )

    treatment_and_score_correlation = user_level_policy_item_context[
        "model_score"
    ].corr(
        logs.item_context["is_treated"]
        + np.random.uniform(1e-10, 1e-9, size=_n_samples * _n_items)
    )

    assert treatment_and_score_correlation < 0.05
