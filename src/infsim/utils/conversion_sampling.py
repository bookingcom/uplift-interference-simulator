from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator


def softmax_along_axis(
    matrix: np.array, temperature: float = 0.1, axis: int = 1
) -> np.array:
    e_x = np.exp((matrix - np.max(matrix, axis=axis, keepdims=True)) / temperature)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def proportion_along_axis(matrix: np.array, axis: int = 1) -> np.array:
    return matrix / np.sum(matrix, axis=axis, keepdims=True)


def sample_one_proportional_to_values_per_row(
    matrix: np.array, rng: Generator
) -> np.array:
    # Generate a random number for each row
    random_values = rng.random(matrix.shape[0])

    # Compute cumulative sums along rows and compare with random values to get column indices
    cumulative_sums = np.cumsum(matrix, axis=1)
    indices = (cumulative_sums > random_values[:, None]).argmax(axis=1)

    # Create a one-hot encoded output matrix
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(matrix.shape[0]), indices] = 1

    return one_hot


def product_conversion_probability(
    attractiveness: np.array, conversion_scaling_factor: float
) -> np.array:
    # Artefact of the transformation of pandas to numpy, all missing values are -inf
    attractiveness[np.isneginf(attractiveness)] = 0

    # We are interested in the probability that a user will not convert to any item
    prob_of_not_conversion = np.product(
        1 - (attractiveness * conversion_scaling_factor), axis=1
    )
    conversion_probability = 1 - prob_of_not_conversion
    return conversion_probability


def max_conversion_probability(attractiveness: np.array) -> np.array:
    return attractiveness.max(axis=1)


def exp_decay_conversion_probability(
    attractiveness: np.array, exp_decay: float = 0.5
) -> np.array:
    attractiveness[np.isneginf(attractiveness)] = 0

    return np.sum(
        exp_decay ** (np.argsort(np.argsort(-attractiveness, axis=1), axis=1) + 1)
        * attractiveness,
        axis=1,
    )


def proportional_exp_decay_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    exp_decay: float = 0.5,
    rng: Optional[Generator] = None,
):
    conversion_probability_func = partial(
        exp_decay_conversion_probability,
        exp_decay=exp_decay,
    )

    return assembled_conversion_sampler(
        user_context,
        item_context,
        conversion_probability_func,
        proportion_along_axis,
        rng,
    )


def softmax_exp_decay_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    temperature: float = 0.1,
    exp_decay: float = 0.5,
    rng: Optional[Generator] = None,
):
    relative_conversion_func = partial(softmax_along_axis, temperature=temperature)
    conversion_probability_func = partial(
        exp_decay_conversion_probability,
        exp_decay=exp_decay,
    )

    return assembled_conversion_sampler(
        user_context,
        item_context,
        conversion_probability_func,
        relative_conversion_func,
        rng,
    )


def softmax_product_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    temperature: float,
    conversion_scaling_factor: float = 1,
    rng: Optional[Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relative_conversion_func = partial(softmax_along_axis, temperature=temperature)
    conversion_probability_func = partial(
        product_conversion_probability,
        conversion_scaling_factor=conversion_scaling_factor,
    )

    return assembled_conversion_sampler(
        user_context,
        item_context,
        conversion_probability_func,
        relative_conversion_func,
        rng,
    )


def softmax_max_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    temperature: float,
    rng: Optional[Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    relative_conversion_func = partial(softmax_along_axis, temperature=temperature)

    return assembled_conversion_sampler(
        user_context,
        item_context,
        max_conversion_probability,
        relative_conversion_func,
        rng,
    )


def proportional_max_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    rng: Optional[Generator] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return assembled_conversion_sampler(
        user_context,
        item_context,
        max_conversion_probability,
        proportion_along_axis,
        rng,
    )


def proportional_product_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    conversion_scaling_factor: float = 1,
    rng: Optional[Generator] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conversion_probability_func = partial(
        product_conversion_probability,
        conversion_scaling_factor=conversion_scaling_factor,
    )
    return assembled_conversion_sampler(
        user_context,
        item_context,
        conversion_probability_func,
        proportion_along_axis,
        rng,
    )


def assembled_conversion_sampler(
    user_context: pd.DataFrame,
    item_context: pd.DataFrame,
    conversion_probability_function: callable,
    relative_probability_function: callable,
    rng: Optional[Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_context = user_context.drop(
        columns=["_conversion_probability"], errors="ignore"
    )
    item_context = item_context.drop(
        columns=["_relative_attractiveness", "is_converted"], errors="ignore"
    )

    if rng is None:
        rng = np.random.default_rng(None)

    # Convert item_context to numpy matrix for efficiency
    item_context["_pd2np_idx"] = item_context.groupby("user_id").cumcount()
    attractiveness = item_context.pivot(
        index="user_id", columns="_pd2np_idx", values="_treatment_attractiveness"
    ).fillna(-np.inf)
    np_attractiveness = attractiveness.to_numpy()

    # Find conversion probability per user
    np_conversion_probability = conversion_probability_function(np_attractiveness)
    pd_conversion_probability = attractiveness.reset_index()[["user_id"]]
    pd_conversion_probability["_conversion_probability"] = np_conversion_probability
    user_context = pd.merge(
        user_context, pd_conversion_probability, on="user_id", how="left"
    )

    _relative_attractiveness = relative_probability_function(np_attractiveness)

    _np_converted_items = sample_one_proportional_to_values_per_row(
        _relative_attractiveness, rng
    )
    _converted_users = rng.binomial(
        1, np_conversion_probability, size=len(np_conversion_probability)
    )

    # Safely join conversion back
    _pd_relative_attractiveness = attractiveness.copy()
    _pd_relative_attractiveness[_pd_relative_attractiveness.columns] = (
        _relative_attractiveness
    )

    item_context = pd.merge(
        item_context,
        pd.melt(
            _pd_relative_attractiveness.reset_index(),
            id_vars=["user_id"],
            value_name="_relative_attractiveness",
        ),
        on=["user_id", "_pd2np_idx"],
        how="left",
    )

    _pd_converted_items = attractiveness.copy()
    _pd_converted_items[_pd_converted_items.columns] = (
        _np_converted_items * _converted_users[:, np.newaxis]
    ).astype(int)

    item_context = pd.merge(
        item_context,
        pd.melt(
            _pd_converted_items.reset_index(),
            id_vars=["user_id"],
            value_name="is_converted",
        ),
        on=["user_id", "_pd2np_idx"],
        how="left",
    )

    del item_context["_pd2np_idx"]

    return user_context, item_context
