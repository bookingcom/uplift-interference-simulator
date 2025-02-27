import re

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from infsim.utils.context_sampler import UniformContextSampler, PandasContextIterator


def test_context_sampler_samples_user_and_item_context_with_specified_shapes():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context, item_context = sampler.sample(_n_samples)

    assert len(item_context) == _n_samples * _n_items
    assert len(user_context) == _n_samples
    assert (
        len([col for col in item_context.columns if "item_feat" in col])
        == _n_item_features
    )
    assert (
        len([col for col in user_context.columns if "user_feat" in col])
        == _n_user_features
    )


def test_context_sampler_samples_dataframe_with_join_key():
    _n_samples = 3
    _n_items = 2
    _n_user_features = 2
    _n_item_features = 3

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    actual_user_context, actual_item_context = sampler.sample(_n_samples)

    expected_user_context = pd.DataFrame(
        [
            {"user_id": 0, "user_feat_1": 0.7, "user_feat_2": 0.7},
            {"user_id": 1, "user_feat_1": 0.4, "user_feat_2": 0.1},
            {"user_id": 2, "user_feat_1": 0.8, "user_feat_2": 1.0},
        ]
    )

    expected_item_context = pd.DataFrame(
        [
            {"user_id": 0, "item_feat_1": 0.7, "item_feat_2": 0.6, "item_feat_3": 0.8},
            {"user_id": 0, "item_feat_1": 0.8, "item_feat_2": 0.8, "item_feat_3": 0.6},
            {"user_id": 1, "item_feat_1": 0.1, "item_feat_2": 0.4, "item_feat_3": 0.8},
            {"user_id": 1, "item_feat_1": 0.5, "item_feat_2": 0.2, "item_feat_3": 0.4},
            {"user_id": 2, "item_feat_1": 0.3, "item_feat_2": 0.6, "item_feat_3": 1.0},
            {"user_id": 2, "item_feat_1": 0.9, "item_feat_2": 0.1, "item_feat_3": 0.9},
        ]
    )

    assert_frame_equal(
        expected_user_context, actual_user_context, check_exact=False, atol=0.1
    )
    assert_frame_equal(
        expected_item_context, actual_item_context, check_exact=False, atol=0.1
    )


def test_context_sampler_increments_user_index_when_new_samples_are_generated():
    _n_samples = 2
    _n_items = 2
    _n_user_features = 2
    _n_item_features = 3

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )

    actual_first_user_context, actual_first_item_context = sampler.sample(_n_samples)
    actual_second_user_context, actual_second_item_context = sampler.sample(_n_samples)

    expected_first_user_context = pd.DataFrame([{"user_id": 0}, {"user_id": 1}])
    expected_second_user_context = pd.DataFrame([{"user_id": 2}, {"user_id": 3}])

    assert_frame_equal(
        expected_first_user_context, actual_first_user_context[["user_id"]]
    )
    assert_frame_equal(
        expected_second_user_context, actual_second_user_context[["user_id"]]
    )

    expected_first_item_context = pd.DataFrame(
        [{"user_id": 0}, {"user_id": 0}, {"user_id": 1}, {"user_id": 1}]
    )
    expected_second_item_context = pd.DataFrame(
        [{"user_id": 2}, {"user_id": 2}, {"user_id": 3}, {"user_id": 3}]
    )

    assert_frame_equal(
        expected_first_item_context, actual_first_item_context[["user_id"]]
    )
    assert_frame_equal(
        expected_second_item_context, actual_second_item_context[["user_id"]]
    )


def test_pandas_context_sampler_is_consistent_given_same_seed():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context_1, item_context_1 = sampler.sample(_n_samples)

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context_2, item_context_2 = sampler.sample(_n_samples)

    assert_frame_equal(user_context_1, user_context_2)
    assert_frame_equal(item_context_1, item_context_2)


def test_pandas_context_sampler_is_different_given_different_seed():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context_1, item_context_1 = sampler.sample(_n_samples)

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=37
    )
    user_context_2, item_context_2 = sampler.sample(_n_samples)

    del user_context_1["user_id"]
    del user_context_2["user_id"]
    del item_context_1["user_id"]
    del item_context_2["user_id"]

    assert np.not_equal(user_context_1.to_numpy(), user_context_2.to_numpy()).all()
    assert np.not_equal(item_context_1.to_numpy(), item_context_2.to_numpy()).all()


def test_pandas_context_iterator_iterates_over_provided_dataset():
    user_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"],
        data=[(1, "a"), (2, "b"), (3, "c"), (4, "d")],
    )

    item_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"],
        data=[(1, "f"), (2, "g"), (2, "g2"), (3, "h"), (4, "e")],
    )

    context_iterator = PandasContextIterator(user_context, item_context)

    first_sample_user_context, first_sample_item_context = context_iterator.sample(2)
    second_sample_user_context, second_sample_item_context = context_iterator.sample(2)

    assert_frame_equal(first_sample_user_context, user_context[0:2])
    assert_frame_equal(first_sample_item_context, item_context[0:3])
    assert_frame_equal(second_sample_user_context, user_context[2:4])
    assert_frame_equal(
        second_sample_item_context, item_context[3:5].reset_index(drop=True)
    )


def test_pandas_context_iterator_joins_using_provided_join_key():
    user_context = pd.DataFrame(
        columns=["user_id_column", "item_feat_1"],
        data=[(1, "a"), (2, "b"), (3, "c"), (4, "d")],
    )

    item_context = pd.DataFrame(
        columns=["user_id_column", "item_feat_1"],
        data=[(1, "f"), (2, "g"), (2, "h"), (4, "e")],
    )

    context_iterator = PandasContextIterator(
        user_context, item_context, join_key="user_id_column"
    )

    sample_user_context, sample_item_context = context_iterator.sample(2)

    assert_frame_equal(sample_user_context, user_context[0:2])
    assert_frame_equal(sample_item_context, item_context[0:3])


def test_pandas_context_iterator_throws_exception_if_user_has_no_matching_items():
    user_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"], data=[(1, "a"), (2, "b")]
    )

    item_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"], data=[(1, "f"), (4, "e")]
    )

    context_iterator = PandasContextIterator(user_context, item_context)

    with pytest.raises(RuntimeError, match=re.escape("No items for `user_id` {2}")):
        _ = context_iterator.sample(2)


def test_pandas_context_iterator_throws_exception_when_data_has_exhausted():
    # If this functionality is used a lot, we can support wrapping
    user_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"],
        data=[(1, "a"), (2, "b"), (3, "c"), (4, "d")],
    )

    item_context = pd.DataFrame(
        columns=["user_id", "item_feat_1"],
        data=[(1, "f"), (2, "g"), (3, "h"), (4, "e")],
    )

    context_iterator = PandasContextIterator(user_context, item_context)

    _ = context_iterator.sample(2)

    with pytest.raises(
        RuntimeError,
        match=re.escape("Dataset contains only 4 users, requesting user 5."),
    ):
        _ = context_iterator.sample(3)
