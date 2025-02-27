from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class ContextSampler:
    def sample(self, n_samples: int) -> Tuple[np.array, np.array]:
        raise NotImplementedError  #  pragma: no cover


@dataclass
class UniformContextSampler(ContextSampler):
    n_items: int
    n_user_features: int
    n_item_features: int
    cur_idx: int = 0
    seed: Optional[int] = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def sample(self, n_samples: int) -> Tuple[np.array, np.array]:
        new_cur_idx = self.cur_idx + n_samples

        users = {"user_id": np.arange(self.cur_idx, new_cur_idx)}
        users.update(
            {
                f"user_feat_{i+1}": self.rng.uniform(size=n_samples)
                for i in range(self.n_user_features)
            }
        )
        user_context = pd.DataFrame(users)

        items = {
            "user_id": np.repeat(np.arange(self.cur_idx, new_cur_idx), self.n_items)
        }
        items.update(
            {
                f"item_feat_{i+1}": self.rng.uniform(size=n_samples * self.n_items)
                for i in range(self.n_item_features)
            }
        )
        item_context = pd.DataFrame(items)

        self.cur_idx = new_cur_idx

        return user_context, item_context


@dataclass
class PandasContextIterator(ContextSampler):
    """
    If user and item context is generated externally from the framework, this class allows to stream a pandas dataframe
    into a base environment. While streaming, the number of returned rows cannot exceed the size of `user_context`.

    user_context: A dataframe containing a join_key and one or multiple features prefixed with user_feat_
    item_context: A dataframe containing a join_key and one or multiple features prefixed with item_feat_
    join_key: A column name that can be matched between the user_context and item_context dataframes.
        Each entry in user_context and item_context should have at least one joinable row.
    """

    user_context: pd.DataFrame
    item_context: pd.DataFrame
    join_key: str = "user_id"
    cur_idx: int = 0

    def sample(self, n_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        new_cur_idx = self.cur_idx + n_samples

        if new_cur_idx > len(self.user_context):
            raise RuntimeError(
                f"Dataset contains only {len(self.user_context)} users, requesting user {new_cur_idx}."
            )

        user_context_sample = self.user_context[self.cur_idx : new_cur_idx]

        item_context_sample = pd.merge(
            user_context_sample[self.join_key],
            self.item_context,
            on=self.join_key,
            how="inner",
        )

        missing_user_ids = set(user_context_sample[self.join_key]) - set(
            item_context_sample[self.join_key]
        )
        if missing_user_ids:
            raise RuntimeError(f"No items for `{self.join_key}` {missing_user_ids}")

        self.cur_idx = new_cur_idx

        return user_context_sample, item_context_sample
