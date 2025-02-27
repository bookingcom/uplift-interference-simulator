import re
from functools import partial

import numpy as np
import pytest

from infsim.environments.base_simulator import BaseEnvironment
from infsim.policies.user_policies import user_level_control_policy
from infsim.utils.attractiveness import user_item_attractiveness
from infsim.utils.context_sampler import UniformContextSampler
from infsim.utils.conversion_sampling import softmax_max_conversion_sampler
from infsim.utils.evaluation import (
    create_buckets_from_scores,
    compute_incremental_metrics,
)


def test_bucketize_creates_equal_sized_buckets_when_possible():
    scores = np.array([1, 2, 9, 10, 5, 6, 7, 8, 3, 4])

    actual_buckets = create_buckets_from_scores(scores, n_buckets=5)

    expected_buckets = np.array([4, 4, 0, 0, 2, 2, 1, 1, 3, 3])
    assert (actual_buckets == expected_buckets).all()


def test_bucketize_creates_different_sized_buckets_when_duplicate_scores_exist():
    scores = np.array([1, 2, 2, 2, 3, 3, 4, 5, 6, 7])

    actual_buckets = create_buckets_from_scores(scores, n_buckets=5)

    expected_buckets = np.array([4, 3, 3, 3, 2, 2, 1, 1, 0, 0])
    assert (actual_buckets == expected_buckets).all()


def test_bucketize_raises_if_not_all_buckets_can_be_created():
    scores = np.array([1, 1, 2, 2, 3, 4])

    with pytest.raises(
        RuntimeError, match=re.escape("Less scores than number of specified buckets")
    ):
        _ = create_buckets_from_scores(scores, n_buckets=5)


def test_bucketize_raises_if_two_buckets_are_more_than_set_percent_different_in_size():
    scores = np.array([1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Relative bucket size exceeds threshold of 1.1x (biggest bucket is 2.0x larger than smallest bucket)"
        ),
    ):
        _ = create_buckets_from_scores(scores, n_buckets=5, max_relative_size=1.1)


def test_bucketize_does_not_raise_if_two_buckets_are_within_set_percent_different_in_size():
    scores = np.array([1, 2, 9, 10, 5, 6, 7, 8, 3, 4])

    actual_buckets = create_buckets_from_scores(
        scores, n_buckets=5, max_relative_size=1.1
    )

    expected_buckets = np.array([4, 4, 0, 0, 2, 2, 1, 1, 3, 3])
    assert (actual_buckets == expected_buckets).all()


def test_ground_truth_incremental_metric_computation_returns_consistent_shape():
    _n_items = 10
    _n_samples = 1000
    _n_item_features = 11
    _n_user_features = 12
    _n_buckets = 10

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

    logs = env.step(_n_samples)

    # We use random values for model scores instead of the ones produced by the policy (which are all .5).
    model_scores = np.random.uniform(size=(_n_samples * _n_items))
    logs.item_context["model_scores"] = model_scores

    assigned_buckets = create_buckets_from_scores(
        logs.item_context["model_scores"], n_buckets=_n_buckets
    )

    incremental_metrics = compute_incremental_metrics(env, logs, assigned_buckets)

    for metric in incremental_metrics.keys():
        assert len(incremental_metrics[metric]) == _n_buckets + 1


def test_ground_truth_incremental_metric_incr_metrics_start_at_zero_and_increment_based_on_non_treatment():
    _n_items = 10
    _n_samples = 1000
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

    logs = env.step(_n_samples)

    # We use random values for model scores instead of the ones produced by the policy (which are all .5).
    model_scores = np.random.uniform(size=(_n_samples * _n_items))
    logs.model_scores = model_scores

    assigned_buckets = create_buckets_from_scores(logs.model_scores, n_buckets=10)

    incremental_metrics = compute_incremental_metrics(env, logs, assigned_buckets)

    assert (
        incremental_metrics["expected_incr_conversions"]
        == incremental_metrics["expected_conversions"]
        - incremental_metrics["expected_conversions"][0]
    ).all()
    assert (
        incremental_metrics["expected_incr_profit"]
        == incremental_metrics["expected_incr_profit"]
        - incremental_metrics["expected_incr_profit"][0]
    ).all()


def test_ground_truth_incremental_metric_treats_bucket_only_if_in_agreement_with_optional_treatment_vector():
    _n_items = 10
    _n_samples = 10
    _n_item_features = 11
    _n_user_features = 12
    _n_buckets = 10

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

    logs = env.step(_n_samples)

    # We use random values for model scores instead of the ones produced by the policy (which are all .5).
    model_scores = np.repeat(np.random.uniform(size=(_n_samples)), 10).reshape(10, 10)
    logs.item_context["model_scores"] = model_scores.ravel()

    # only treat the first 10 items
    treatment_mask = np.concatenate(
        [np.ones(shape=(10, 2)), np.zeros(shape=(10, 8))], axis=1
    )

    assigned_buckets = create_buckets_from_scores(
        logs.item_context["model_scores"], n_buckets=_n_buckets
    )

    incremental_metrics = compute_incremental_metrics(
        env, logs, assigned_buckets, treatment_mask=treatment_mask.ravel()
    )

    for i, treatments in enumerate(incremental_metrics["_treatments"]):
        assert treatments.reshape(10, 10)[:, 2:].sum() == 0
        assert treatments.reshape(10, 10)[:, :2].sum() == 2 * i
