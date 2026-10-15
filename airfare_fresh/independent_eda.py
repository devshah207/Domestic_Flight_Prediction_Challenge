#!/usr/bin/env python3
"""Independent, descriptive EDA. Reads labeled train only; fits no predictor.

All target summaries are diagnostics, never features. Any learned duration or
categorical mapping suggested by this report must be fitted inside each model
training partition. Test/sample/preserved submission files are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/airfare-fresh-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/airfare-fresh-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_TRAIN = "/workspace/attachments/085aeb37-413d-42c9-bb16-baafd968488d/train.csv"
DEFAULT_OUT = "/workspace/airfare_fresh/eda"


def as_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=as_json, allow_nan=False) + "\n")


def main(train=DEFAULT_TRAIN, out=DEFAULT_OUT):
    path, out = Path(train), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(path)
    expected = {"id", "airline", "flight", "source_city", "departure_time",
                "stops", "arrival_time", "destination_city", "flight_class",
                "duration", "days_left", "price"}
    assert set(x) == expected and "price" in x
    raw_cols = [c for c in x if c not in {"id", "price"}]
    x["prefix"] = x.flight.str.extract(r"^([^ ]+)")[0]
    x["flight_number"] = x.flight.str.extract(r"(\d+)$")[0].astype(int)
    x["route"] = x.source_city + ">" + x.destination_city
    x["duration_missing"] = x.duration.isna()
    x["airline_missing"] = x.airline.isna()
    x["booking_bucket"] = pd.cut(x.days_left, [0, 7, 14, 21, 28, 35, 42, 49, 57],
                                 labels=["1–7", "8–14", "15–21", "22–28", "29–35", "36–42", "43–49", "50–57"])
    sst = float(((x.price - x.price.mean()) ** 2).sum())
    descriptive = {
        "input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(x), "raw_columns": list(expected),
        "categorical_counts": {c: int(x[c].nunique(dropna=True)) for c in raw_cols if x[c].dtype == "object"},
        "missing_counts": {c: int(x[c].isna().sum()) for c in raw_cols},
        "price_quantiles": {str(q): float(x.price.quantile(q)) for q in [0, .01, .25, .5, .75, .95, .99, 1]},
        "duration_quantiles": {str(q): float(x.duration.quantile(q)) for q in [0, .01, .25, .5, .75, .95, .99, 1]},
        "days_range": [int(x.days_left.min()), int(x.days_left.max())],
        "class_counts": x.flight_class.value_counts().to_dict(),
        "raw_negative_or_zero_duration": int((x.duration <= 0).sum()),
        "raw_nonpositive_price": int((x.price <= 0).sum()),
        "self_routes": int((x.source_city == x.destination_city).sum()),
        "id_used_as_feature": False, "kaggle_test_read": False,
        "scope": "Descriptive training-only EDA; no model fitted and no feature selected from validation scores."
    }

    # The categorical labels recover each other only at appropriate granularity.
    prefix_airline = x.dropna(subset=["airline"]).groupby("prefix").airline.agg(["nunique", "count", "first"])
    prefix_airline.to_csv(out / "prefix_airline_mapping_diagnostic.csv")
    descriptive["prefix_airline_max_labels"] = int(prefix_airline["nunique"].max())
    descriptive["missing_airline_prefixes_without_observed_label"] = sorted(set(x.loc[x.airline_missing, "prefix"]) - set(prefix_airline.index))

    # Conditional missingness is a diagnostic; do not interpret small differences
    # as mechanisms, or duplicate these diagnostics as fitted features.
    missing_rows = []
    for cols in [["flight_class"], ["prefix"], ["route"], ["booking_bucket"], ["flight_class", "prefix"]]:
        for key, g in x.groupby(cols, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            missing_rows.append({"grouping": "+".join(cols), "key": "|".join(map(str, key)),
                                 "n": len(g), "airline_missing_rate": g.airline_missing.mean(),
                                 "duration_missing_rate": g.duration_missing.mean(),
                                 "mean_price": g.price.mean()})
    missing_df = pd.DataFrame(missing_rows)
    missing_df.to_csv(out / "conditional_missingness.csv", index=False)
    missing_table = pd.crosstab(x.airline_missing, x.duration_missing)
    descriptive["joint_missingness_counts"] = {str(k): v for k, v in missing_table.to_dict().items()}

    # Prices vary by class and carrier, not just cabin class or a single airline.
    p = x.groupby(["flight_class", "prefix"]).price.agg(
        n="size", mean="mean", std="std", median="median", minimum="min", maximum="max",
        p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9))
    p.to_csv(out / "price_by_class_prefix.csv")
    dep_rows = []
    for key, g in x.groupby(["flight_class", "prefix"]):
        if len(g) < 30:
            continue
        close = g.loc[g.days_left <= 7, "price"]
        far = g.loc[g.days_left >= 28, "price"]
        observed = g.dropna(subset=["duration"])
        dep_rows.append({"flight_class": key[0], "prefix": key[1], "n": len(g),
                         "days_price_spearman": spearmanr(g.days_left, g.price).statistic,
                         "duration_price_spearman": spearmanr(observed.duration, observed.price).statistic if len(observed) > 2 else np.nan,
                         "flight_suffix_price_spearman": spearmanr(g.flight_number, g.price).statistic,
                         "close_n": len(close), "far_n": len(far),
                         "close_far_median_ratio": close.median() / far.median() if len(close) and len(far) else np.nan})
    dependencies = pd.DataFrame(dep_rows)
    dependencies.to_csv(out / "conditional_dependence.csv", index=False)
    booking = x.groupby(["flight_class", "prefix", "booking_bucket"], observed=True).price.agg(["size", "median", "mean"])
    booking.to_csv(out / "booking_price_curves.csv")

    # Sparse interactions: thresholds report effective observations, not merely
    # number of unique category levels. These are target-free support counts.
    sparsity = []
    for cols in [["flight"], ["flight", "route"], ["flight", "route", "flight_class"],
                 ["prefix", "route", "flight_class"],
                 ["prefix", "route", "flight_class", "stops"],
                 ["prefix", "route", "flight_class", "departure_time", "arrival_time"]]:
        counts = x.groupby(cols, dropna=False).size()
        support = x.groupby(cols, dropna=False).price.transform("size")
        sparsity.append({"keys": "+".join(cols), "n_groups": len(counts), "median_group_n": counts.median(),
                         "singleton_groups": int((counts == 1).sum()),
                         "rows_in_groups_n_le_2": int((support <= 2).sum()),
                         "rows_in_groups_n_le_5": int((support <= 5).sum()),
                         "rows_in_groups_n_le_10": int((support <= 10).sum())})
    sparsity = pd.DataFrame(sparsity)
    sparsity.to_csv(out / "interaction_sparsity.csv", index=False)

    # Same raw features with disagreeing labels are not necessarily errors;
    # missing date/layover/capacity can create unresolved outcome variation.
    x["raw_duplicate_group"] = pd.util.hash_pandas_object(x[raw_cols], index=False).astype(str)
    conflicts = x.groupby("raw_duplicate_group").agg(
        n=("price", "size"), prices=("price", "nunique"), min_price=("price", "min"),
        max_price=("price", "max"), mean_price=("price", "mean"),
        duration_missing=("duration_missing", "first"),
        flight_class=("flight_class", "first"), prefix=("prefix", "first"), flight=("flight", "first"),
        route=("route", "first"), days_left=("days_left", "first"))
    conflicts["range"] = conflicts.max_price - conflicts.min_price
    conflicts.query("prices > 1").sort_values("range", ascending=False).to_csv(out / "exact_duplicate_conflicts.csv")
    duplicate_mean = x.groupby("raw_duplicate_group").price.transform("mean")
    duplicate_sse = float(((x.price - duplicate_mean) ** 2).sum())
    descriptive["duplicates"] = {
        "raw_feature_duplicate_excess_rows": int(x.duplicated(raw_cols).sum()),
        "exact_rows_including_target_excess": int(x.duplicated(raw_cols + ["price"]).sum()),
        "raw_groups_with_multiple_prices": int((conflicts.prices > 1).sum()),
        "rows_in_conflicting_groups": int(conflicts.loc[conflicts.prices > 1, "n"].sum()),
        "max_within_identical_features_price_range": float(conflicts["range"].max()),
        "within_duplicate_group_sse_over_total_target_sst": duplicate_sse / sst,
        "interpretation": "An in-sample ambiguity diagnostic, not a validated ceiling on model performance; no observations removed."
    }

    # Duration in hours is often a schedule property; route and schedule resolve
    # flight reuse. Remaining 24-hour ambiguity cautions against overconfident fill.
    duration_rows, detailed_duration = [], {}
    for label, cols in [
        ("flight", ["flight"]),
        ("flight+route", ["flight", "route"]),
        ("flight+route+schedule", ["flight", "route", "departure_time", "arrival_time", "stops"]),
        ("prefix+route+schedule", ["prefix", "route", "departure_time", "arrival_time", "stops"]),
    ]:
        d = x.groupby(cols).duration.agg(n="count", std="std", minimum="min", maximum="max", median="median")
        d["range"] = d.maximum - d.minimum
        eligible = d.loc[d.n >= 3]
        duration_rows.append({"grouping": label, "n_groups": len(d), "groups_n_ge_3": len(eligible),
                              "median_std_hours": eligible["std"].median(),
                              "p90_std_hours": eligible["std"].quantile(.9),
                              "groups_range_gt_1hr": int((eligible["range"] > 1).sum()),
                              "groups_range_about_24hr": int(np.isclose(eligible["range"], 24, atol=.02).sum()),
                              "max_range_hours": eligible["range"].max()})
        detailed_duration[label] = d
        d.to_csv(out / (label.replace("+", "_") + "_duration_variation.csv"))
    durations = pd.DataFrame(duration_rows)
    durations.to_csv(out / "duration_group_consistency.csv", index=False)
    descriptive["one_stop_duration_under_3h"] = int(((x.stops == "one") & (x.duration < 3)).sum())
    descriptive["top_1pct_target_sst_share"] = float(((x.loc[x.price >= x.price.quantile(.99), "price"] - x.price.mean()) ** 2).sum() / sst)
    save_json(out / "data_summary.json", descriptive)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False)
    for ax, klass in zip(axes, ["Economy", "Business"]):
        t = p.loc[klass].query("n >= 30").sort_values("median")
        y = np.arange(len(t))
        ax.hlines(y, t.p10 / 1000, t.p90 / 1000, color="#85b8d0", lw=5)
        ax.scatter(t["median"] / 1000, y, color="#123b52", zorder=3, s=35)
        ax.set_yticks(y, [f"{v} (n={int(n)})" for v, n in zip(t.index, t.n)])
        ax.set(xlabel="Training price, thousands; median and 10th–90th percentiles", title=klass)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Cabin and flight prefix jointly define distinct price regimes")
    fig.tight_layout()
    fig.savefig(out / "class_prefix_price_regimes.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, klass, selected in zip(axes, ["Economy", "Business"], [["6E", "AI", "QP", "IX", "SG"], ["6E", "AI", "EK", "EY", "UL", "TG"]]):
        for prefix in selected:
            vals = booking.loc[(klass, prefix)]
            vals = vals.loc[vals["size"] >= 10]
            ax.plot(vals.index.astype(str), vals["median"] / 1000, marker="o", ms=3, label=prefix)
        ax.set(xlabel="Days until departure (binned)", ylabel="Median training price, thousands", title=klass)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(ncol=3, frameon=False)
        ax.grid(alpha=.2)
    fig.suptitle("Booking curves differ by carrier and cabin; global monotonicity is unsupported")
    fig.tight_layout()
    fig.savefig(out / "booking_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = []
    for i, (label, d) in enumerate(detailed_duration.items()):
        vals = d.loc[d.n >= 3, "std"].clip(lower=.001)
        axes[0].boxplot(vals, positions=[i], showfliers=False, widths=.5)
        labels.append(label.replace("+", "\n"))
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(4), labels)
    axes[0].set(ylabel="Within-group duration SD (hours, log scale)", title="Adding route and schedule resolves flight reuse")
    axes[0].grid(axis="y", alpha=.2)
    m = missing_df.query("grouping == 'flight_class+prefix' and n >= 100").sort_values("key")
    y = np.arange(len(m))
    axes[1].barh(y - .17, m.duration_missing_rate * 100, height=.34, label="Duration")
    axes[1].barh(y + .17, m.airline_missing_rate * 100, height=.34, label="Airline")
    axes[1].set_yticks(y, m.key.str.replace("Business|", "B ", regex=False).str.replace("Economy|", "E ", regex=False))
    axes[1].set(xlabel="Missing rows (%)", title="Missingness by cabin and prefix")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "duration_and_missingness.png")
    plt.close(fig)

    recommendation = {
        "mandatory": ["Exclude id.", "Keep original-price R2 as metric.", "Keep exact raw feature duplicate groups together in all splits.",
                      "Learn imputers/encodings inside each training partition; descriptive EDA tables are not feature lookup tables.",
                      "No neural networks, unlimited-depth trees, external labels, or Kaggle-test inspection."],
        "variant_1_raw_minimal": {
            "categorical": ["flight", "prefix", "source_city", "destination_city", "departure_time", "arrival_time", "stops", "flight_class"],
            "numeric": ["duration", "days_left"],
            "missing": "Native duration NaN when supported; stable categorical missing marker. Airline is redundant with prefix in observed training rows.",
            "hypothesis": "Preserve raw signal while avoiding a large bundle of redundant sparse interactions and arbitrary flight-number ordering."
        },
        "variant_2_parsimonious_interactions": {
            "add_to_minimal": ["route", "prefix_class"],
            "hypothesis": "Route has only 30 levels and cabin/prefix price regimes are large; these compact features expose important decision structure.",
            "avoid_by_default": ["flight_route_class_time", "dense target histories", "rare-group price encodings"]
        },
        "variant_3_duration_fill_ablation": {
            "scope": "Only if native missing performs poorly; retain raw duration and duration_missing.",
            "fold_local_hierarchy": ["flight+route+departure+arrival+stops", "flight+route", "prefix+route+stops", "route+stops", "stops", "global"],
            "regularization": "Require minimum observed support and fall back for uncertain groups; same schedules can differ by 24 hours.",
            "hypothesis": "Recovered duration may be useful in large AI/6E business groups; imputation is not an automatic improvement."
        },
        "not_supported": ["Global monotone price decrease with days_left.", "Dropping high prices as erroneous labels.",
                          "Treating numeric flight suffix as an intrinsically ordered economic quantity.",
                          "Removing all conflicting duplicate rows or assuming all duplicate conflicts are annotation mistakes.",
                          "Using every possible sparse category interaction."]
    }
    save_json(out / "feature_recommendations.json", recommendation)

    sparseline = sparsity.loc[sparsity["keys"] == "flight+route+flight_class"].iloc[0]
    native_missing = missing_df.query("grouping == 'flight_class+prefix' and n >= 100")
    table_cols = ["grouping", "groups_n_ge_3", "median_std_hours", "p90_std_hours", "groups_range_about_24hr", "max_range_hours"]
    def format_cell(value):
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    durations_table = "| " + " | ".join(table_cols) + " |\n| " + " | ".join(["---"] * len(table_cols)) + " |\n"
    durations_table += "\n".join("| " + " | ".join(map(format_cell, row)) + " |" for row in durations[table_cols].itertuples(index=False, name=None))
    report = f"""# Independent training-only EDA

Scope: read only the supplied labeled training CSV ({len(x):,} rows). No model fitted, no Kaggle test/sample or historical prediction files opened. This is fresh descriptive work, not a rerun of the historical feature builder. Reproduce with `python /workspace/airfare_fresh/independent_eda.py`.

## What the raw observations say

- There are {x.flight.nunique():,} flight identifiers, {x.prefix.nunique()} prefixes, {x.route.nunique()} directional routes and two cabins. Prices range from {x.price.min():,} to {x.price.max():,}; 99th percentile is {x.price.quantile(.99):,.0f}. Airline is missing in {x.airline_missing.mean():.2%} and duration in {x.duration_missing.mean():.2%}.
- Every prefix with an observed airline maps to one airline label. All missing-airline prefixes have an observed training mapping. AI and IX share the same airline label, so prefix carries finer information. A compact raw pipeline can use prefix directly instead of guessing missing airlines. Any learned label mapping must remain fold-local.
- Business median prices differ sharply by prefix: AI {p.loc[('Business','AI'),'median']:,.0f}, 6E {p.loc[('Business','6E'),'median']:,.0f}, UL {p.loc[('Business','UL'),'median']:,.0f}, EK {p.loc[('Business','EK'),'median']:,.0f}, EY {p.loc[('Business','EY'),'median']:,.0f}. QP has only 30 business observations (versus 830 economy); fine business interactions there are sparse.
- Duration–price association changes sign by prefix within Business: Spearman AI +0.632 and 6E +0.773, but EK −0.241 and EY −0.382. A global linear duration effect or a simple duration-based fare correction is unjustified. A joint tree model can represent this heterogeneity.
- Booking horizon is not globally monotonic. Business days–price Spearman is UL −0.371 but TG +0.132; Economy AI and 6E are approximately zero, IX +0.117, QP +0.109. Close/far median ratios are descriptive and mix routes, so they are not causal booking-discount estimates. Preserve continuous days_left; do not impose global monotonicity based on business intuition.
- Missingness varies modestly across the large cabin/prefix groups: duration {native_missing.duration_missing_rate.min():.1%}–{native_missing.duration_missing_rate.max():.1%}, airline {native_missing.airline_missing_rate.min():.1%}–{native_missing.airline_missing_rate.max():.1%}. This is compatible with broadly scattered missingness; it does not establish missing-at-random. Missing flags are reasonable low-cost options, not proven signals.

## Duration structure and ambiguity

{durations_table}

Flight numbers are reused on routes or schedules. Route/schedule conditioning sharply reduces within-group duration variation. Even fully specified schedules sometimes differ by 24 hours, and one one-stop row reports less than three hours. Do not automatically delete these as errors: layovers and day rollover can be hidden. Native missing duration is a clean baseline; any hierarchical fill should retain missingness and avoid giving tiny/noisy groups unjustified confidence. Duration diagnostics above use observed values only, no target.

## Duplicate and sparse-group risks

There are {descriptive['duplicates']['raw_feature_duplicate_excess_rows']} extra rows with exactly the same raw features excluding id, versus {descriptive['duplicates']['exact_rows_including_target_excess']} extra fully identical labeled rows. {descriptive['duplicates']['raw_groups_with_multiple_prices']} feature groups contain different prices, covering {descriptive['duplicates']['rows_in_conflicting_groups']} rows; the largest observed price range within identical features is {descriptive['duplicates']['max_within_identical_features_price_range']:,.0f}. Their in-sample within-group squared variation is {descriptive['duplicates']['within_duplicate_group_sse_over_total_target_sst']:.6%} of total target SST. These collisions show hidden variables or label ambiguity, not grounds to remove rows or labels. Keep exact raw feature groups together during validation.

Flight×route×class has {int(sparseline.n_groups):,} groups, with {int(sparseline.rows_in_groups_n_le_5):,} rows in groups of at most five observations. More detailed flight/schedule/class interactions would further fragment rare expensive business cases. Use compact route and prefix×class features before adding an interaction bundle. Numeric flight suffix is a code whose ordering has no guaranteed economic interpretation; preserve categorical flight and treat numeric suffix inclusion as an optional ablation.

No high-price observations were removed or relabeled. Rare expensive tickets are strongly structured by prefix/class; extremeness alone is insufficient evidence of corruption. Any residual/tail diagnosis must come from held-out predictions, not the in-sample summaries here.

## Limited features to test

1. Raw minimal: categorical flight/prefix/source/destination/departure/arrival/stops/class; numeric raw duration and days_left; native missing duration.
2. Compact interactions: add route and prefix_class to the raw minimal set. These are deterministic row-wise transforms, with no fitted cross-row information.
3. Only as a separate ablation: fold-local duration fill with explicit support/uncertainty and missing flag; retain original raw features. Avoid simultaneously changing target, feature bundle and categorical representation.

All feature recommendations remain hypotheses. This report has no CV/model score and does not select a winner. Tune bounded models with inner-only stopping, score on untouched outer folds, and report original-price pooled R2 and train–validation gaps. Do not infer that a low gap proves no overfitting.

## Artifacts

- `data_summary.json`: source hash, shape, target/missingness and duplicate facts.
- `conditional_dependence.csv`, `booking_price_curves.csv`, `price_by_class_prefix.csv`: raw associations and support.
- `interaction_sparsity.csv`: target-free support for proposed groups.
- `duration_group_consistency.csv` and detailed duration tables: schedule reliability.
- `exact_duplicate_conflicts.csv`: conflicting duplicate groups, retained unchanged.
- `feature_recommendations.json`: machine-readable, small feature variants.
- Three PNG figures show fare regimes, booking curves, duration reliability and missingness.
"""
    (out / "TRAIN_ONLY_EDA.md").write_text(report)
    print(json.dumps({"rows": len(x), "output": str(out), "source_sha256": descriptive["sha256"],
                      "duplicates": descriptive["duplicates"], "artifacts": len(list(out.iterdir()))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.train, args.out)
