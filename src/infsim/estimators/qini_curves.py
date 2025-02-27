from typing import Callable, Tuple
from infsim.environments.base_simulator import PolicyLogs

import numpy as np
import pandas as pd


def resample_data(
    user_context: pd.DataFrame, item_context: pd.DataFrame, treatment: np.array
) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    assert (
        item_context.shape[0] == treatment.shape[0]
    ), f"item_context and treatment must have same number of rows: {user_context.shape[0]}, {item_context.shape[0]}, {treatment.shape[0]}"

    # Resampling on user-level (warning: untested!)
    n_users = len(user_context)
    resampled_users = np.random.choice(n_users, size=n_users, replace=True)

    resampled_user_context = user_context.iloc[resampled_users].reset_index(drop=True)
    resampled_user_ids = resampled_user_context.user_id
    resampled_items = item_context.user_id.isin(resampled_user_ids)

    resampled_item_context = item_context[resampled_items].reset_index(drop=True)
    resampled_treatment = treatment[resampled_items]

    return resampled_user_context, resampled_item_context, resampled_treatment


def compute_qini_curve(
    buckets: list,
    logs: PolicyLogs,
    estimator: Callable,
    only_use_converted_cases=True,
    n_bootstraps: int = None,
) -> dict:
    sorted_buckets = sorted(np.unique(buckets))
    sorted_buckets += [max(sorted_buckets) + 1]  # add to treat all as an option

    buckets = buckets.copy()
    user_context = logs.user_context.copy()
    item_context = logs.item_context.copy()

    item_context["revenue"] = (
        logs.price * (logs.commission_percentage / 100) * item_context["is_converted"]
    )
    item_context["cost"] = (
        logs.price
        * logs.discount_percentage
        * item_context["is_treated"]
        * item_context["is_converted"]
    )
    item_context["profit"] = item_context["revenue"] - item_context["cost"]

    if only_use_converted_cases:
        converted_user_ids = item_context[item_context.is_converted == 1][
            "user_id"
        ].unique()
        buckets = buckets[item_context.user_id.isin(converted_user_ids)]
        user_context = user_context[user_context.user_id.isin(converted_user_ids)]
        item_context = item_context[item_context.user_id.isin(converted_user_ids)]

    incr_profit = np.array([])
    incr_conversions = np.array([])
    var_incr_conversions = np.array([])
    var_incr_profit = np.array([])

    for bucket in sorted_buckets:
        treatment_decision = buckets < bucket
        out = estimator(user_context, item_context, treatment_decision)
        incr_conversions = np.append(incr_conversions, out["expected_n_conversions"])
        incr_profit = np.append(incr_profit, out["expected_profit"])

        if n_bootstraps:
            resampled_conversions = np.array([])
            resampled_profit = np.array([])
            for i in range(n_bootstraps):
                resampled_user_context, resampled_item_context, resampled_treatment = (
                    resample_data(user_context, item_context, treatment_decision)
                )
                resampled_out = estimator(
                    resampled_user_context, resampled_item_context, resampled_treatment
                )
                resampled_conversions = np.append(
                    resampled_conversions, resampled_out["expected_n_conversions"]
                )
                resampled_profit = np.append(
                    resampled_profit, resampled_out["expected_profit"]
                )
            var_incr_conversions = np.append(
                var_incr_conversions, np.var(resampled_conversions)
            )
            var_incr_profit = np.append(var_incr_profit, np.var(resampled_profit))
        else:
            var_incr_conversions = np.append(var_incr_conversions, np.nan)
            var_incr_profit = np.append(var_incr_profit, np.nan)

    incr_profit = incr_profit - incr_profit[0]
    incr_conversions = incr_conversions - incr_conversions[0]

    return {
        "expected_incr_profit": incr_profit,
        "expected_incr_conversions": incr_conversions,
        "var_incr_profit": var_incr_profit,
        "var_incr_conversions": var_incr_conversions,
    }
