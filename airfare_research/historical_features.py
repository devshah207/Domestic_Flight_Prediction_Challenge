"""Faithful historical feature builder and sparse OHE, extracted without changes."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = [
    "airline_fixed", "flight", "flight_prefix", "source_city", "destination_city",
    "departure_time", "arrival_time", "stops", "flight_class",
    "route", "route_stops", "route_class", "prefix_class", "dep_arr"
]

NUM_COLS = ["duration_filled", "duration_missing", "days_left", "flight_number", "airline_missing", "stops_numeric"]

OHE_CAT_COLS = [c for c in CAT_COLS if c != "flight"]

LGB_CAT_COLS = [c for c in CAT_COLS if c != "flight"] + ["flight_rare"]

class TrainingOnlyFeatureBuilder:
    """Learns only unsupervised mappings from the labeled training-set FEATURES."""
    def fit(self, x):
        x = x.copy()
        x["flight_prefix"] = x["flight"].astype(str).str.split().str[0]
        observed_airline = x.dropna(subset=["airline"])
        self.prefix_to_airline = observed_airline.groupby("flight_prefix")["airline"].agg(
            lambda s: s.mode().iloc[0]
        ).to_dict()
        observed_duration = x.dropna(subset=["duration"])
        self.duration_global_median = float(observed_duration["duration"].median())
        self.duration_group_tables = []
        # Each hierarchy is chosen from training-only feature consistency, not target.
        keys_with_required_count = [
            (["flight", "source_city", "destination_city", "stops"], 2),
            (["flight", "source_city", "destination_city"], 2),
            (["flight", "stops"], 2),
            (["flight"], 2),
            (["source_city", "destination_city", "stops", "flight_class"], 5),
            (["source_city", "destination_city", "stops"], 5),
            (["source_city", "destination_city"], 5),
            (["stops"], 5),
        ]
        for keys, min_count in keys_with_required_count:
            grp = observed_duration.groupby(keys, dropna=False)["duration"].agg(["median", "count"])
            medians = grp.loc[grp["count"] >= min_count, "median"]
            self.duration_group_tables.append((keys, medians))
        # Group infrequent flight IDs without looking at target prices.
        flight_counts = x["flight"].value_counts(dropna=False)
        self.frequent_flights = set(flight_counts[flight_counts >= 12].index.astype(str))
        return self

    def transform(self, x):
        # Take only feature columns; 'id' is intentionally excluded as mechanical.
        x = x.copy().reset_index(drop=True)
        out = pd.DataFrame(index=x.index)
        for col in ["flight", "source_city", "destination_city", "departure_time",
                    "arrival_time", "stops", "flight_class"]:
            out[col] = x[col].fillna("Unknown").astype(str)
        out["flight_prefix"] = out["flight"].str.split().str[0].fillna("Unknown")
        out["flight_number"] = pd.to_numeric(
            out["flight"].str.extract(r"(\d+)\s*$", expand=False), errors="coerce"
        ).fillna(-1.0).astype(float)
        out["airline_missing"] = x["airline"].isna().astype(int)
        out["airline_fixed"] = x["airline"].fillna(out["flight_prefix"].map(self.prefix_to_airline)).fillna("Unknown").astype(str)
        out["days_left"] = pd.to_numeric(x["days_left"], errors="coerce").fillna(-1.0).astype(float)
        out["stops_numeric"] = out["stops"].map({"zero": 0, "one": 1, "two_or_more": 2}).fillna(-1).astype(float)
        out["duration_missing"] = x["duration"].isna().astype(int)
        current = pd.to_numeric(x["duration"], errors="coerce").astype(float).copy()
        for keys, table in self.duration_group_tables:
            need = current.isna()
            if not need.any():
                break
            if len(keys) == 1:
                new_values = out.loc[need, keys[0]].map(table)
            else:
                idx = pd.MultiIndex.from_frame(out.loc[need, keys])
                new_values = pd.Series(table.reindex(idx).to_numpy(), index=out.index[need])
            current.loc[need] = current.loc[need].fillna(new_values)
        out["duration_filled"] = current.fillna(self.duration_global_median).astype(float)
        out["route"] = out["source_city"] + " -> " + out["destination_city"]
        out["route_stops"] = out["route"] + " / " + out["stops"]
        out["route_class"] = out["route"] + " / " + out["flight_class"]
        out["prefix_class"] = out["flight_prefix"] + " / " + out["flight_class"]
        out["dep_arr"] = out["departure_time"] + " -> " + out["arrival_time"]
        out["flight_rare"] = out["flight"].where(out["flight"].isin(self.frequent_flights), "__RARE_OR_NEW__")
        return out

    def fit_transform(self, x):
        return self.fit(x).transform(x)

def make_ohe_features(train_features):
    # Note: all sklearn estimators below are trained on these precomputed TRAIN-only features.
    # Test is transformed later using frozen OneHotEncoders (no fitting on test).
    prep = ColumnTransformer(
        transformers=[
            ("regular_categories", OneHotEncoder(handle_unknown="ignore", sparse_output=True), OHE_CAT_COLS),
            ("flight_rare", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=12,
                                          sparse_output=True), ["flight"]),
            ("numerical", "passthrough", NUM_COLS),
        ],
        sparse_threshold=1.0,
    )
    return prep, prep.fit_transform(train_features).tocsr().astype(np.float32)
