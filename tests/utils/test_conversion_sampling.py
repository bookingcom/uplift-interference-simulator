import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from infsim.utils.conversion_sampling import (
    softmax_max_conversion_sampler,
    softmax_along_axis,
    softmax_product_conversion_sampler,
    proportional_max_conversion_sampler,
    proportional_product_conversion_sampler,
    softmax_exp_decay_conversion_sampler,
    proportional_exp_decay_conversion_sampler,
)


@pytest.mark.parametrize(
    "input, axis, temperature, expected",
    [
        ([[1, 2, 3], [3, 2, 1]], 1, 1.0, [[0.09, 0.24, 0.67], [0.67, 0.24, 0.09]]),
        ([[1, 2, 3], [3, 2, 1]], 0, 1.0, [[0.12, 0.5, 0.88], [0.88, 0.5, 0.12]]),
        ([[1, 2, 3]], 1, 100.0, [[0.33, 0.33, 0.34]]),
        ([[1, 2, 3]], 1, 0.0001, [[0.0, 0.0, 1.0]]),
    ],
)
def test_softmax_function_on_axis_and_temperature(input, axis, temperature, expected):
    # round for readability
    actual = np.round(
        softmax_along_axis(input, temperature=temperature, axis=axis), decimals=2
    )
    assert np.equal(actual, expected).all()


@pytest.mark.parametrize("temperature", [100, 10, 1, 0.0001])
def test_pandas_softmax_max_conversion_sampler_returns_vector_with_at_most_one_hot(
    temperature,
):
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame([{"user_id": 1}, {"user_id": 2}])

    _, item_context_with_conversions = softmax_max_conversion_sampler(
        user_context, item_context, temperature=temperature
    )

    assert np.logical_or(
        item_context_with_conversions.groupby("user_id")["is_converted"].sum() == 1,
        item_context_with_conversions.groupby("user_id")["is_converted"].sum() == 0,
    ).all()


def test_pandas_softmax_max_conversion_sampler_adds_conversion_and_relative_attractiveness_to_item_context():
    rng = np.random.default_rng(42)

    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame([{"user_id": 1}, {"user_id": 2}])

    true_user_context, item_context_with_conversions = softmax_max_conversion_sampler(
        user_context, item_context, temperature=1, rng=rng
    )

    expected_conversions_attractiveness = pd.DataFrame(
        [
            {
                "user_id": 1,
                "item_id": 1,
                "_treatment_attractiveness": 0.2,
                "_relative_attractiveness": 0.28,
                "is_converted": 0,
            },
            {
                "user_id": 1,
                "item_id": 2,
                "_treatment_attractiveness": 0.3,
                "_relative_attractiveness": 0.32,
                "is_converted": 0,
            },
            {
                "user_id": 1,
                "item_id": 3,
                "_treatment_attractiveness": 0.5,
                "_relative_attractiveness": 0.39,
                "is_converted": 1,
            },
            {
                "user_id": 2,
                "item_id": 4,
                "_treatment_attractiveness": 0.4,
                "_relative_attractiveness": 0.45,
                "is_converted": 0,
            },
            {
                "user_id": 2,
                "item_id": 5,
                "_treatment_attractiveness": 0.6,
                "_relative_attractiveness": 0.54,
                "is_converted": 0,
            },
        ]
    )

    assert_frame_equal(
        expected_conversions_attractiveness,
        item_context_with_conversions,
        check_exact=False,
        atol=0.01,
    )


def test_proportional_max_conversion_sampler_adds_correct_conversion_and_relative_attractiveness_to_item_context():
    rng = np.random.default_rng(42)

    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.5},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
            {"user_id": 2, "item_id": 6, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame([{"user_id": 1}, {"user_id": 2}])

    true_user_context, item_context_with_conversions = (
        proportional_max_conversion_sampler(
            user_context, item_context, temperature=1, rng=rng
        )
    )

    expected_conversions_attractiveness = pd.DataFrame(
        [
            {
                "user_id": 1,
                "item_id": 1,
                "_treatment_attractiveness": 0.5,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
            {
                "user_id": 1,
                "item_id": 2,
                "_treatment_attractiveness": 0.5,
                "_relative_attractiveness": 0.5,
                "is_converted": 1,
            },
            {
                "user_id": 2,
                "item_id": 5,
                "_treatment_attractiveness": 0.6,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
            {
                "user_id": 2,
                "item_id": 6,
                "_treatment_attractiveness": 0.6,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
        ]
    )

    assert_frame_equal(
        expected_conversions_attractiveness,
        item_context_with_conversions,
        check_exact=False,
        atol=0.01,
    )

    expected_user_context = pd.DataFrame(
        [
            {"user_id": 1, "_conversion_probability": 0.5},
            {"user_id": 2, "_conversion_probability": 0.6},
        ]
    )

    assert_frame_equal(
        expected_user_context,
        true_user_context,
        check_exact=False,
        atol=0.01,
    )


def test_proportional_product_conversion_sampler_adds_correct_conversion_and_relative_attractiveness_to_item_context():
    rng = np.random.default_rng(42)

    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.5},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
            {"user_id": 2, "item_id": 6, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame([{"user_id": 1}, {"user_id": 2}])

    true_user_context, item_context_with_conversions = (
        proportional_product_conversion_sampler(user_context, item_context, rng=rng)
    )

    expected_conversions_attractiveness = pd.DataFrame(
        [
            {
                "user_id": 1,
                "item_id": 1,
                "_treatment_attractiveness": 0.5,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
            {
                "user_id": 1,
                "item_id": 2,
                "_treatment_attractiveness": 0.5,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
            {
                "user_id": 2,
                "item_id": 5,
                "_treatment_attractiveness": 0.6,
                "_relative_attractiveness": 0.5,
                "is_converted": 1,
            },
            {
                "user_id": 2,
                "item_id": 6,
                "_treatment_attractiveness": 0.6,
                "_relative_attractiveness": 0.5,
                "is_converted": 0,
            },
        ]
    )

    assert_frame_equal(
        expected_conversions_attractiveness,
        item_context_with_conversions,
        check_exact=False,
        atol=0.01,
    )

    expected_user_context = pd.DataFrame(
        [
            {"user_id": 1, "_conversion_probability": 0.75},
            {"user_id": 2, "_conversion_probability": 0.84},
        ]
    )

    assert_frame_equal(
        expected_user_context,
        true_user_context,
        check_exact=False,
        atol=0.01,
    )


def test_sample_conversion_probability_as_product_of_1_minus_attractiveness_and_adds_to_user_context_dataframe():
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame(
        [{"user_id": 1, "user_feat_1": "a"}, {"user_id": 2, "user_feat_1": "a"}]
    )

    actual_user_context, _ = softmax_product_conversion_sampler(
        user_context, item_context, temperature=0.1
    )

    expected_user_context = pd.DataFrame(
        [
            {
                "user_id": 1,
                "user_feat_1": "a",
                "_conversion_probability": 1 - ((1 - 0.2) * (1 - 0.3) * (1 - 0.5)),
            },
            {
                "user_id": 2,
                "user_feat_1": "a",
                "_conversion_probability": 1 - ((1 - 0.4) * (1 - 0.6)),
            },
        ]
    )
    assert_frame_equal(expected_user_context, actual_user_context)


@pytest.mark.parametrize(
    "exp_decay, conversion_sampler",
    [
        [0.2, softmax_exp_decay_conversion_sampler],
        [0.5, softmax_exp_decay_conversion_sampler],
        [0.2, proportional_exp_decay_conversion_sampler],
        [0.5, proportional_exp_decay_conversion_sampler],
    ],
)
def test_sample_conversion_probability_as_exp_decay_applies_set_decay_parameter(
    exp_decay, conversion_sampler
):
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.6},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.4},
        ]
    )

    user_context = pd.DataFrame(
        [{"user_id": 1, "user_feat_1": "a"}, {"user_id": 2, "user_feat_1": "a"}]
    )

    actual_user_context, _ = conversion_sampler(
        user_context, item_context, exp_decay=exp_decay
    )

    expected_user_context = pd.DataFrame(
        [
            {
                "user_id": 1,
                "user_feat_1": "a",
                "_conversion_probability": np.sum(
                    0.2 * exp_decay**3 + 0.3 * exp_decay**2 + 0.5 * exp_decay**1
                ),
            },
            {
                "user_id": 2,
                "user_feat_1": "a",
                "_conversion_probability": np.sum(
                    0.6 * exp_decay**1 + 0.4 * exp_decay**2
                ),
            },
        ]
    )
    assert_frame_equal(expected_user_context, actual_user_context)


def test_sample_conversion_probability_as_product_allows_scaling_of_individual_attractiveness_to_reduce_total_prob():
    SCALE = 0.5
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame(
        [{"user_id": 1, "user_feat_1": "a"}, {"user_id": 2, "user_feat_1": "a"}]
    )

    actual_user_context, _ = softmax_product_conversion_sampler(
        user_context, item_context, temperature=0.1, conversion_scaling_factor=SCALE
    )

    expected_user_context = pd.DataFrame(
        [
            {
                "user_id": 1,
                "user_feat_1": "a",
                "_conversion_probability": 1
                - ((1 - 0.2 * SCALE) * (1 - 0.3 * SCALE) * (1 - 0.5 * SCALE)),
            },
            {
                "user_id": 2,
                "user_feat_1": "a",
                "_conversion_probability": 1 - ((1 - 0.4 * SCALE) * (1 - 0.6 * SCALE)),
            },
        ]
    )
    assert_frame_equal(expected_user_context, actual_user_context)


def test_sample_conversion_probability_as_maximum_of_attractiveness_and_adds_to_user_context_dataframe():
    item_context = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
        ]
    )

    user_context = pd.DataFrame(
        [{"user_id": 1, "user_feat_1": "a"}, {"user_id": 2, "user_feat_1": "a"}]
    )

    actual_user_context, _ = softmax_max_conversion_sampler(
        user_context, item_context, temperature=0.1
    )

    expected_user_context = pd.DataFrame(
        [
            {"user_id": 1, "user_feat_1": "a", "_conversion_probability": 0.5},
            {"user_id": 2, "user_feat_1": "a", "_conversion_probability": 0.6},
        ]
    )
    assert_frame_equal(expected_user_context, actual_user_context)


def test_sample_conversion_probability_works_if_users_are_not_ordered_correctly():
    item_context = pd.DataFrame(
        [
            {"user_id": 2, "item_id": 4, "_treatment_attractiveness": 0.4},
            {"user_id": 2, "item_id": 5, "_treatment_attractiveness": 0.6},
            {"user_id": 1, "item_id": 1, "_treatment_attractiveness": 0.2},
            {"user_id": 1, "item_id": 2, "_treatment_attractiveness": 0.3},
            {"user_id": 1, "item_id": 3, "_treatment_attractiveness": 0.5},
        ]
    )

    user_context = pd.DataFrame(
        [{"user_id": 2, "user_feat_1": "a"}, {"user_id": 1, "user_feat_1": "a"}]
    )

    actual_user_context, _ = softmax_max_conversion_sampler(
        user_context, item_context, temperature=0.1
    )

    expected_user_context = pd.DataFrame(
        [
            {"user_id": 2, "user_feat_1": "a", "_conversion_probability": 0.6},
            {"user_id": 1, "user_feat_1": "a", "_conversion_probability": 0.5},
        ]
    )
    assert_frame_equal(expected_user_context, actual_user_context)
