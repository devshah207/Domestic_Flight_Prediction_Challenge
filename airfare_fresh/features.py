"""Small, fold-local feature sets for the independent airfare study.

This module never opens a data file, consumes targets, or fits a model.  Fit a
new ``TabularFeatures`` instance on each training partition, then use that
instance for its validation/inference partition.  ``id`` and ``price`` are
deliberately excluded even if supplied in the input frame.

``minimal`` retains the ten raw predictors, adds the flight prefix and two
missingness flags, and fills airline names from a training-only prefix map.
``context`` adds only route, prefix-by-class, numeric stops, and raw duration.
Its ``duration`` column is filled from training-only schedule/group medians;
``duration_raw`` still carries the original numeric values and missing values.

For XGBoost and LightGBM, categorical dtypes use a fitted vocabulary and unseen
categories become missing.  For CatBoost, categorical columns are strings and
unseen strings are retained for the library's native categorical handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TabularFeatures(TransformerMixin, BaseEstimator):
    """Serializable feature transformer with no target-dependent statistics.

    Parameters
    ----------
    mode : {"minimal", "context"}
        ``minimal`` preserves numerical missing values for native treatment.
        ``context`` also provides fold-local imputed duration and three small
        contextual features, while preserving the unfilled duration.
    family : {"xgb", "lgb", "cat"}
        Categorical output representation.  Supply ``enable_categorical=True``
        for XGBoost and ``cat_features=transformer.cat_columns_`` for CatBoost.

    Notes
    -----
    ``fit`` must be called inside each relevant inner/outer training split.
    Airline prefix modes and duration medians never use evaluation rows.
    The output has a RangeIndex to prevent accidental index alignment errors.
    Numeric columns are float32.  Every transform returns the same column order.
    """

    RAW_COLUMNS = (
        "airline",
        "flight",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "flight_class",
        "duration",
        "days_left",
    )
    RAW_CATEGORICAL = RAW_COLUMNS[:8]
    MISSING = "__MISSING__"
    DURATION_MIN_COUNT = 2
    DURATION_GROUPS = (
        ("flight", "route", "stops", "departure_time", "arrival_time"),
        ("flight", "route", "stops"),
        ("route", "stops"),
    )

    def __init__(self, mode: str = "minimal", family: str = "xgb") -> None:
        self.mode = mode
        self.family = family

    @classmethod
    def _string(cls, series: pd.Series) -> pd.Series:
        """Normalize whitespace without learning or changing category meaning."""
        out = series.astype("string").str.strip()
        return out.mask(out.eq(""), pd.NA).fillna(cls.MISSING).astype(object)

    def _validate_input(self, raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("TabularFeatures requires a pandas DataFrame.")
        missing = sorted(set(self.RAW_COLUMNS).difference(raw.columns))
        if missing:
            raise ValueError(f"Required predictor columns are missing: {missing}")
        if raw.columns.has_duplicates:
            raise ValueError("Input columns must have unique names.")
        return raw.loc[:, self.RAW_COLUMNS].reset_index(drop=True).copy()

    def _raw_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        frame = self._validate_input(raw)
        for column in self.RAW_CATEGORICAL:
            frame[column] = self._string(frame[column])
        frame["flight_prefix"] = self._string(
            frame["flight"].str.extract(r"^([A-Za-z0-9]+)", expand=False)
        )
        # Prefix extraction of the missing marker must remain missing, too.
        frame.loc[frame["flight"].eq(self.MISSING), "flight_prefix"] = self.MISSING
        for column in ("duration", "days_left"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        frame["airline_missing"] = frame["airline"].eq(self.MISSING).astype("float32")
        frame["duration_missing"] = frame["duration"].isna().astype("float32")
        return frame

    @staticmethod
    def _deterministic_mode(values: pd.Series) -> str:
        counts = values.value_counts()
        return str(sorted(counts.index[counts.eq(counts.max())])[0])

    def _context_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_airline = frame["airline"].eq(self.MISSING)
        replacement = frame["flight_prefix"].map(self.airline_by_prefix_)
        frame.loc[missing_airline, "airline"] = replacement[missing_airline].fillna(
            self.MISSING
        )
        if self.mode == "context":
            frame["route"] = frame["source_city"] + " > " + frame["destination_city"]
            frame["prefix_class"] = frame["flight_prefix"] + " | " + frame["flight_class"]
            frame["stops_numeric"] = frame["stops"].map(
                {"zero": 0.0, "one": 1.0, "two_or_more": 2.0}
            )
            frame["duration_raw"] = frame["duration"]
        return frame

    def fit(self, raw: pd.DataFrame, y: Any = None) -> "TabularFeatures":
        """Learn only feature-only mappings from the supplied training rows."""
        if self.mode not in {"minimal", "context"}:
            raise ValueError("mode must be 'minimal' or 'context'.")
        if self.family not in {"xgb", "lgb", "cat"}:
            raise ValueError("family must be 'xgb', 'lgb', or 'cat'.")
        frame = self._raw_frame(raw)
        if frame.empty:
            raise ValueError("Cannot fit features on an empty training frame.")
        self.n_features_in_ = len(self.RAW_COLUMNS)
        self.feature_names_in_ = np.asarray(self.RAW_COLUMNS, dtype=object)
        known_airline = frame["airline"].ne(self.MISSING) & frame["flight_prefix"].ne(
            self.MISSING
        )
        self.airline_by_prefix_ = (
            frame.loc[known_airline]
            .groupby("flight_prefix", observed=True)["airline"]
            .agg(self._deterministic_mode)
            .to_dict()
        )
        frame = self._context_frame(frame)
        self.duration_maps_ = []
        self.duration_global_ = float("nan")
        if self.mode == "context":
            duration = frame["duration"]
            if duration.notna().any():
                self.duration_global_ = float(duration.median())
            for columns in self.DURATION_GROUPS:
                summary = frame.groupby(list(columns), observed=True)["duration"].agg(
                    ["median", "count"]
                )
                valid = summary.loc[summary["count"].ge(self.DURATION_MIN_COUNT), "median"]
                self.duration_maps_.append((columns, valid.to_dict()))
        self.cat_columns_ = list(self.RAW_CATEGORICAL) + ["flight_prefix"]
        if self.mode == "context":
            self.cat_columns_ += ["route", "prefix_class"]
        self.feature_columns_ = list(frame.columns)
        self.numeric_columns_ = [c for c in self.feature_columns_ if c not in self.cat_columns_]
        self.category_levels_ = {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in self.cat_columns_
        }
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen mappings; never update statistics from these rows."""
        check_is_fitted(self, ["feature_columns_", "airline_by_prefix_", "category_levels_"])
        frame = self._context_frame(self._raw_frame(raw))
        if self.mode == "context":
            filled = frame["duration"].copy()
            for columns, mapping in self.duration_maps_:
                if filled.isna().any():
                    keys = pd.Series(
                        list(frame.loc[:, list(columns)].itertuples(index=False, name=None)),
                        index=frame.index,
                        dtype=object,
                    )
                    filled = filled.fillna(keys.map(mapping))
            frame["duration"] = filled.fillna(self.duration_global_)
        for column in self.cat_columns_:
            if self.family == "cat":
                frame[column] = frame[column].astype(str)
            else:
                frame[column] = pd.Categorical(
                    frame[column], categories=self.category_levels_[column], ordered=False
                )
        for column in self.numeric_columns_:
            frame[column] = frame[column].astype("float32")
        return frame.loc[:, self.feature_columns_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, ["feature_columns_"])
        return np.asarray(self.feature_columns_, dtype=object)

