import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal, assert_frame_equal

from infsim.utils.attractiveness import (
    base_user_item_attractiveness,
    attractiveness_from_pandas,
    user_item_attractiveness,
)
from infsim.utils.context_sampler import UniformContextSampler

ATTRACTIVENESS_COLUMNS = [
    "_base_attractiveness",
    "_treatment_attractiveness",
    "_treatment_attractiveness_delta",
]


def test_attractiveness_score_changes_when_user_context_changes():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    user_context = np.random.uniform(size=(_n_samples, _n_user_features))
    user_context_changed = np.random.uniform(size=(_n_samples, _n_user_features))
    item_context = np.random.uniform(size=(_n_samples, _n_items, _n_item_features))

    attractiveness = base_user_item_attractiveness(user_context, item_context)
    attractiveness_user_context_changed = base_user_item_attractiveness(
        user_context_changed, item_context
    )

    assert np.not_equal(attractiveness, attractiveness_user_context_changed).all()


def test_attractiveness_score_changes_when_item_context_changes():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    user_context = np.random.uniform(size=(_n_samples, _n_user_features))
    item_context = np.random.uniform(size=(_n_samples, _n_items, _n_item_features))
    item_context_changed = np.random.uniform(
        size=(_n_samples, _n_items, _n_item_features)
    )

    attractiveness = base_user_item_attractiveness(user_context, item_context)
    attractiveness_item_context_changed = base_user_item_attractiveness(
        user_context, item_context_changed
    )

    assert np.not_equal(attractiveness, attractiveness_item_context_changed).all()


def test_attractiveness_score_is_identical_when_context_is_identical():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    user_context = np.random.uniform(size=(1, _n_user_features))
    item_context = np.random.uniform(size=(1, 1, _n_item_features))

    user_context = np.tile(user_context, reps=(_n_samples, 1))
    item_context = np.tile(item_context, reps=(_n_samples, _n_items, 1))

    attractiveness = base_user_item_attractiveness(user_context, item_context)

    assert np.all(attractiveness == attractiveness[0, 0])


def test_attractiveness_score_is_similar_when_context_is_similar():
    _n_samples = 2
    _n_items = 1
    _n_user_features = 11
    _n_item_features = 12

    user_context = np.random.uniform(size=(1, _n_user_features))
    item_context = np.random.uniform(size=(1, 1, _n_item_features))
    item_context = np.tile(item_context, reps=(_n_samples, _n_items, 1))

    similar_user_contexts = np.vstack([user_context, user_context + 0.01])
    similar_attractiveness = base_user_item_attractiveness(
        similar_user_contexts, item_context, seed=42
    ).ravel()
    similar_context_diff = abs(similar_attractiveness[0] - similar_attractiveness[1])

    different_user_contexts = np.vstack([user_context, user_context + 0.2])
    different_attractiveness = base_user_item_attractiveness(
        different_user_contexts, item_context, seed=42
    ).ravel()
    different_context_diff = abs(
        different_attractiveness[0] - different_attractiveness[1]
    )

    assert (similar_context_diff * 10) < different_context_diff


def test_attractiveness_score_is_different_when_context_is_different():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12

    user_context = np.random.uniform(size=(_n_samples, _n_user_features))
    item_context = np.random.uniform(size=(_n_samples, _n_items, _n_item_features))

    attractiveness = base_user_item_attractiveness(user_context, item_context)

    assert np.unique(attractiveness).size == attractiveness.size


def test_attractiveness_score_is_between_0_and_1():
    _n_samples = 1000
    _n_items = 150
    _n_user_features = 110
    _n_item_features = 120

    user_context = np.random.uniform(size=(_n_samples, _n_user_features))
    item_context = np.random.uniform(size=(_n_samples, _n_items, _n_item_features))

    attractiveness = base_user_item_attractiveness(user_context, item_context)

    assert np.all((attractiveness >= 0) & (attractiveness <= 1))


@pytest.mark.parametrize(
    "_n_user_features, _n_item_features",
    [
        (5, 5),
        (5, 10),
        (100, 50),
        (500, 1000),
        (1000, 1000),
    ],
)
def test_attractiveness_scale_stays_consistent_regardless_of_nr_of_features(
    _n_user_features, _n_item_features
):
    _n_samples = 1000
    _n_items = 10

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)
    item_context["is_treated"] = 0
    item_context = user_item_attractiveness(user_context, item_context, 0.08, seed=42)

    assert item_context._base_attractiveness.mean() < 0.2
    assert item_context._base_attractiveness.mean() > 0.05


def test_attractiveness_has_only_50_percent_items_with_delta_above_0():
    _n_samples = 10000
    _n_items = 10

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=2,
        n_item_features=2,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)
    item_context["is_treated"] = 0
    item_context = user_item_attractiveness(user_context, item_context, 0.08, seed=42)

    assert round((item_context._treatment_attractiveness_delta > 0).mean(), 2) == 0.5


def test_attractiveness_only_changes_for_treated_properties():
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

    item_context["is_treated"] = np.zeros(shape=(_n_samples * _n_items))
    attractiveness_not_treated = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )._treatment_attractiveness

    item_context["is_treated"] = np.random.binomial(1, 0.5, (_n_samples * _n_items))
    attractiveness_treated = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )._treatment_attractiveness

    # ensure there is no difference between the attractiveness of identical treatment
    assert (
        (attractiveness_treated - attractiveness_not_treated)
        * (1 - item_context["is_treated"])
        == 0
    ).all()
    assert (
        ((attractiveness_treated - attractiveness_not_treated) >= 0)
        | (1 - item_context["is_treated"])
    ).all()


def test_attractiveness_after_treatment_is_between_0_and_1():
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

    item_context["is_treated"] = np.ones(shape=(_n_samples * _n_items))
    attractiveness_treated = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )._treatment_attractiveness

    assert np.all((attractiveness_treated >= 0) & (attractiveness_treated <= 1))


def test_attractiveness_when_treated_increases_with_higher_discount():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage_low = 0.08
    _discount_percentage_high = 0.30

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)
    item_context["is_treated"] = np.ones(shape=(_n_samples * _n_items))

    attractiveness_treated_high = user_item_attractiveness(
        user_context, item_context, _discount_percentage_high, seed=42
    )._treatment_attractiveness
    attractiveness_treated_low = user_item_attractiveness(
        user_context, item_context, _discount_percentage_low, seed=42
    )._treatment_attractiveness

    assert attractiveness_treated_high.sum() > attractiveness_treated_low.sum()


def test_attractiveness_when_not_treated_stays_constant_with_higher_discount():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage_low = 0.08
    _discount_percentage_high = 0.30

    context_sampler = UniformContextSampler(
        n_items=_n_items,
        n_user_features=_n_user_features,
        n_item_features=_n_item_features,
        seed=42,
    )

    user_context, item_context = context_sampler.sample(_n_samples)
    item_context["is_treated"] = np.zeros(shape=(_n_samples * _n_items))

    attractiveness_treated_high = user_item_attractiveness(
        user_context, item_context, _discount_percentage_high, seed=42
    )._treatment_attractiveness
    attractiveness_treated_low = user_item_attractiveness(
        user_context, item_context, _discount_percentage_low, seed=42
    )._treatment_attractiveness

    assert attractiveness_treated_high.sum() == attractiveness_treated_low.sum()


def test_attractiveness_from_pandas_adds_attractiveness_and_treatment_to_item_context():
    user_context = pd.DataFrame(columns=["user_id"], data=[(1), (2), (3), (4)])

    item_context = pd.DataFrame(
        columns=[
            "user_id",
            "_base_attractiveness",
            "_treatment_attractiveness_delta",
            "is_treated",
        ],
        data=[(1, 0.1, 0.2, 1), (2, 0.2, 0.3, 1), (3, 0.3, 0.4, 0), (4, 0.3, 0.2, 0)],
    )

    item_context_with_attractiveness = attractiveness_from_pandas(
        user_context, item_context
    )

    assert_series_equal(
        item_context_with_attractiveness._base_attractiveness,
        pd.Series([0.1, 0.2, 0.3, 0.3]),
        check_names=False,
    )
    assert_series_equal(
        item_context_with_attractiveness._treatment_attractiveness_delta,
        pd.Series([0.2, 0.3, 0.4, 0.2]),
        check_names=False,
    )
    assert_series_equal(
        item_context_with_attractiveness._treatment_attractiveness,
        pd.Series([0.3, 0.5, 0.3, 0.3]),
        check_names=False,
    )
    assert_series_equal(
        item_context_with_attractiveness.is_treated,
        item_context["is_treated"],
        check_names=False,
    )


def test_attractiveness_score_is_added_to_item_context():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context, item_context = sampler.sample(_n_samples)

    item_context["is_treated"] = 1

    for col in ATTRACTIVENESS_COLUMNS:
        assert col not in item_context

    actual_item_context = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )

    for col in ATTRACTIVENESS_COLUMNS:
        assert col in actual_item_context


def test_attractiveness_score_is_identical_with_same_treatment():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context, item_context = sampler.sample(_n_samples)

    item_context["is_treated"] = 0

    item_context_one = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )
    item_context_two = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )

    assert_frame_equal(
        item_context_one[ATTRACTIVENESS_COLUMNS],
        item_context_two[ATTRACTIVENESS_COLUMNS],
        check_exact=False,
    )


def test_attractiveness_score_does_not_change_based_based_on_other_contexts_in_logs():
    _n_samples = 2
    _n_items = 9
    _n_user_features = 2
    _n_item_features = 2
    _discount_percentage = 0.08

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context, item_context = sampler.sample(_n_samples)

    item_context["is_treated"] = 0

    item_context_zero = user_item_attractiveness(
        user_context[user_context["user_id"] == 0],
        item_context[item_context["user_id"] == 0],
        _discount_percentage,
        seed=42,
    )
    item_context_one = user_item_attractiveness(
        user_context[user_context["user_id"] == 1],
        item_context[item_context["user_id"] == 1],
        _discount_percentage,
        seed=42,
    )

    item_context_all = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )

    assert_frame_equal(
        item_context_zero, item_context_all[item_context_all["user_id"] == 0]
    )
    assert_frame_equal(
        item_context_one, item_context_all[item_context_all["user_id"] == 1]
    )


def test_attractiveness_score_is_higher_with_more_treatments():
    _n_samples = 1000
    _n_items = 15
    _n_user_features = 11
    _n_item_features = 12
    _discount_percentage = 0.08

    sampler = UniformContextSampler(
        _n_items, _n_user_features, _n_item_features, seed=42
    )
    user_context, item_context = sampler.sample(_n_samples)

    item_context["is_treated"] = 0

    item_context_one = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )

    item_context["is_treated"] = 1
    item_context_two = user_item_attractiveness(
        user_context, item_context, _discount_percentage, seed=42
    )

    assert (
        item_context_one["_treatment_attractiveness"].sum()
        < item_context_two["_treatment_attractiveness"].sum()
    )
    assert (
        item_context_one["_treatment_attractiveness"]
        <= item_context_two["_treatment_attractiveness"]
    ).all()
