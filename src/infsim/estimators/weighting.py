import itertools
import math
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit


class WeightingEstimator:
    _weight_level = "user"

    def __init__(self, save_weights: bool = False):
        if type(self) == WeightingEstimator:
            raise Exception("WeightingEstimator must be subclassed.")

        self.save_weights = save_weights
        self.weights = None
        self.weight_level = "user"

        assert self._weight_level in ["item", "user"]

    def estimate_uplift(
        self,
        user_context: pd.DataFrame,
        item_context: pd.DataFrame,
        treatment_decision: np.array,
    ) -> dict:
        item_context = item_context.copy()
        item_context["treatment_decision"] = treatment_decision
        item_context["is_matched"] = (
            item_context["is_treated"] == item_context["treatment_decision"]
        )

        weights = self.compute_weights(item_context)
        if self._weight_level == "user":
            conversions = item_context.groupby("user_id")["is_converted"].sum().values
        else:
            conversions = item_context["is_converted"].values
        assert np.array_equal(weights.shape, conversions.shape), "shapes do not match"

        if self.save_weights:
            self.weights = weights

        out = {
            "expected_conversion_rate": (weights * conversions).sum()
            / len(user_context),
            "expected_n_conversions": (weights * conversions).sum(),
        }

        out["variance_conversion_rate"] = (
            (weights * conversions - out["expected_conversion_rate"]) ** 2
        ).sum() / (len(user_context) - 1)

        if "profit" in item_context:
            if self._weight_level == "user":
                profit = item_context.groupby("user_id")["profit"].sum().values
            else:
                profit = item_context["profit"].values
            assert np.array_equal(weights.shape, profit.shape), "shapes do not match"
            out["expected_profit"] = (weights * profit).sum()

        return out

    def set_save_weights(self, save_weights: bool) -> None:
        assert isinstance(save_weights, bool), "save_weights must be a boolean"
        self.save_weights = save_weights

    def get_weights(self):
        if self.save_weights is not True:
            raise ValueError(
                "save_weights needs to be True to get weighted conversion."
            )
        if self.save_weights is None:
            raise ValueError(
                "No weighted conversion has been calculated yet. Call estimate_uplift first."
            )
        else:
            return self.weights

    def compute_weights(self, item_context: pd.DataFrame) -> dict:
        raise NotImplementedError("Must be implemented in subclass.")


class ItemLevelWeighting(WeightingEstimator):
    _weight_level = "item"

    def __init__(self, save_weights: bool = False):
        super().__init__(save_weights=save_weights)
        self.propensity = 1 / 2

    def compute_weights(self, item_context):
        is_matched = item_context["is_matched"]
        weights = is_matched / self.propensity
        return weights.values

    def compute_weighted_conversion(self, group) -> float:
        raise NotImplementedError

    def compute_weighted_profit(self, group) -> float:
        raise NotImplementedError


class IPW(WeightingEstimator):
    _weight_level = "user"

    def __init__(self, save_weights: bool = False):
        super().__init__(save_weights=save_weights)

    def compute_weights(self, item_context):
        grouped = item_context.groupby("user_id")
        n_items = grouped.size()
        is_matched = np.array(
            [np.all(matches[1]) for matches in grouped["is_matched"]]
        ).squeeze()
        propensity = 1 / (2**n_items)
        weights = is_matched / propensity

        return weights.values

    def compute_weighted_conversion(self, group) -> float:
        if (group["is_treated"] == group["treatment_decision"]).all():
            propensity = 1 / (2 ** len(group))
            return (1 / propensity) * group["is_converted"].sum()
        else:
            return 0

    def compute_weighted_profit(self, group) -> float:
        if (group["is_treated"] == group["treatment_decision"]).all():
            propensity = 1 / (2 ** len(group))
            return (1 / propensity) * group["profit"].sum()
        else:
            return 0


class AdditiveIPW(WeightingEstimator):
    _weight_level = "user"

    def __init__(self, propensity: float = 1 / 2, save_weights: bool = False):
        super().__init__(save_weights=save_weights)
        self.propensity = propensity

    def compute_weights(self, item_context: pd.DataFrame) -> dict:
        grouped = item_context.groupby("user_id")
        n_items = grouped.size()
        matched_sizes = grouped["is_matched"].sum()
        weights = (1 / self.propensity) * matched_sizes - n_items + 1

        return weights.values


class MultiplicativeIPW(WeightingEstimator):
    _weight_level = "user"

    def __init__(
        self,
        propensity: float = 1 / 2,
        max_cardinality: float = 2,
        save_weights: bool = False,
    ):
        super().__init__(save_weights=save_weights)
        self.max_cardinality = max_cardinality
        self.propensity = propensity

    @staticmethod
    @njit
    def fast_weight_calculation(
        n, mt, propensity_factor, power_sets, power_set_lengths
    ):
        valid_subset = []
        for i in range(len(power_sets)):
            if (
                power_set_lengths[i] == 0
                or np.max(power_sets[i][: power_set_lengths[i]]) < n
            ):
                valid_subset.append(i)

        weights = np.zeros(len(valid_subset))
        for i, subset_idx in enumerate(valid_subset):
            subset_length = power_set_lengths[subset_idx]
            subset = power_sets[subset_idx][:subset_length]
            prod = 1.0
            for j in range(subset_length):
                prod *= propensity_factor * mt[subset[j]] - 1
            weights[i] = prod

        return 1 + np.sum(weights)

    def compute_weights_optimized(
        self, n_items: np.array, matched_treatments: np.array
    ) -> np.array:
        max_n = np.max(n_items)

        # Pre-compute power sets
        power_sets_list = list(
            itertools.chain.from_iterable(
                itertools.combinations(range(max_n), r)
                for r in range(1, min(self.max_cardinality, max_n) + 1)
            )
        )
        max_length = max(len(subset) for subset in power_sets_list)

        power_sets = np.full((len(power_sets_list), max_length), -1, dtype=np.int64)
        power_set_lengths = np.zeros(len(power_sets_list), dtype=np.int64)

        for i, subset in enumerate(power_sets_list):
            power_sets[i, : len(subset)] = subset
            power_set_lengths[i] = len(subset)

        propensity_factor = 1 / self.propensity

        # Vectorized computation using Numba
        weights = np.array(
            [
                self.fast_weight_calculation(
                    n, mt, propensity_factor, power_sets, power_set_lengths
                )
                for n, mt in zip(n_items, matched_treatments)
            ]
        )

        return weights

    def compute_weights(self, item_context: pd.DataFrame) -> Tuple[np.array, np.array]:
        grouped = item_context.groupby("user_id")

        n_items = grouped.size().values
        is_matched = grouped["is_matched"].apply(lambda x: x.values)

        # Compute weights
        weights = self.compute_weights_optimized(n_items, is_matched)

        return weights


class FractionalIPW(WeightingEstimator):
    _weight_level = "item"

    def __init__(self, save_weights: bool = False):
        super().__init__(save_weights=save_weights)

    def compute_weights(self, item_context):
        grouped = item_context.groupby("user_id")
        n_items = grouped.size()
        n_treated_observed = grouped["is_treated"].sum()
        n_treated_decision = grouped["treatment_decision"].sum()

        fraction_matched = n_treated_observed == n_treated_decision
        evaluated_fraction = n_treated_decision / n_items
        fraction_probability = np.array(
            [math.comb(ni, gs) for ni, gs in zip(n_items, n_treated_decision)]
        ) / (2**n_items)

        new_columns = pd.DataFrame(
            {
                "user_id": fraction_matched.index,
                "fraction_matched": fraction_matched.values,
                "evaluated_fraction": evaluated_fraction.values,
                "fraction_probability": fraction_probability.values,
                "n_items": n_items.values,
            }
        )
        item_context = pd.merge(item_context, new_columns, on="user_id", how="left")

        # propensity is equal to Pr(W_ij|\bar{W_i})Pr(\bar{W_i}) where \bar{W_i} is the fraction of treated for user i
        propensity = np.where(
            item_context["treatment_decision"],
            item_context["evaluated_fraction"],
            1 - item_context["evaluated_fraction"],
        )
        propensity *= item_context["fraction_probability"]
        # propensity *= 1 / item_context['n_items']  # not sure why but this factor helps...
        weights = (
            item_context["is_matched"] * item_context["fraction_matched"] / propensity
        )

        return weights.values


def compute_combination_coefficient(
    mean_biased, mean_unbiased, var_biased, var_unbiased, covariance
) -> float:
    # see equation (5) in https://arxiv.org/pdf/2205.10467)
    bias = mean_biased - mean_unbiased
    # print(bias, var_biased, var_unbiased, covariance)
    return (var_unbiased - covariance) / (
        bias**2 + var_unbiased + var_biased - 2 * covariance
    )


class CombinedIPW(WeightingEstimator):
    _weight_level = "user"

    def __init__(
        self, ipw_unbiased: WeightingEstimator, ipw_biased: WeightingEstimator
    ):
        super().__init__()
        self.ipw_unbiased = ipw_unbiased
        self.ipw_biased = ipw_biased

        self.ipw_unbiased.set_save_weights(save_weights=True)
        self.ipw_biased.set_save_weights(save_weights=True)

    def estimate_uplift(
        self,
        user_context: pd.DataFrame,
        item_context: pd.DataFrame,
        treatment_decision: np.array,
    ) -> dict:
        res_biased = self.ipw_biased.estimate_uplift(
            user_context, item_context, treatment_decision
        )
        res_unbiased = self.ipw_unbiased.estimate_uplift(
            user_context, item_context, treatment_decision
        )

        conversions = item_context.groupby("user_id")["is_converted"].sum()
        weights_biased = self.ipw_biased.get_weights()
        weights_unbiased = self.ipw_unbiased.get_weights()

        assert conversions.shape == weights_biased.shape == weights_unbiased.shape

        weighted_conversions_unbiased = conversions * weights_unbiased
        weighted_conversions_biased = conversions * weights_biased

        deviations_biased = (
            weighted_conversions_biased - weighted_conversions_biased.mean()
        )
        deviations_unbiased = (
            weighted_conversions_unbiased - weighted_conversions_unbiased.mean()
        )
        covariance = (deviations_biased * deviations_unbiased).mean()

        coef = compute_combination_coefficient(
            res_biased["expected_conversion_rate"],
            res_unbiased["expected_conversion_rate"],
            res_biased["variance_conversion_rate"],
            res_unbiased["variance_conversion_rate"],
            covariance,
        )

        out = {
            "expected_conversion_rate": coef * res_biased["expected_conversion_rate"]
            + (1 - coef) * res_unbiased["expected_conversion_rate"],
            "expected_n_conversions": coef * res_biased["expected_n_conversions"]
            + (1 - coef) * res_unbiased["expected_n_conversions"],
            "combination_coef": coef,
        }

        if "profit" in item_context:
            out["expected_profit"] = (
                coef * res_biased["expected_profit"]
                + (1 - coef) * res_unbiased["expected_profit"]
            )

        return out

    def compute_weighted_conversion(self, group) -> float:
        pass

    def compute_weighted_profit(self, group) -> float:
        pass
