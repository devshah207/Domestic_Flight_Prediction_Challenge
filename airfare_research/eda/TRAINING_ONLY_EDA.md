# Training-only structural audit

The only data source is labeled `train.csv` (SHA256 `6b0c9793817dc3e1b229ecea9ebbef14eef34452abf05a6f708d197cf6d8cc07`). No Kaggle test, sample submission, previous prediction CSV, external data, or leaderboard data was read by this script.

This audit is descriptive and hypothesis-generating. Broad EDA using training labels is **not independent proof** of predictive improvement. Every learned imputer, category map, statistic, and encoding used in a predictive comparison must be fitted within that comparison's training partition. No predictive model was fitted in this script.

## Reconciled data and ambiguity

There are 15,589 training observations. Target mean 29,816.70, median 10,183.00, SD 39,127.89, range 3,801–323,177. Airline missing 3,089; duration missing 3,095. Every observed prefix has exactly one airline label, so prefix fill is feature-redundancy recovery, not a new target signal.

```json
{
  "raw_duplicates": {
    "groups": 15275,
    "repeated_groups": 310,
    "repeated_rows": 624,
    "redundant_rows": 314,
    "conflicting_groups": 35,
    "conflicting_rows": 71,
    "observed_within_key_sse": 450764660.5,
    "observed_within_key_sse_fraction_total_variation": 1.8888026571273782e-05
  },
  "airline_canonical_duplicates": {
    "groups": 15119,
    "repeated_groups": 454,
    "repeated_rows": 924,
    "redundant_rows": 470,
    "conflicting_groups": 48,
    "conflicting_rows": 99,
    "observed_within_key_sse": 688957273.6666667,
    "observed_within_key_sse_fraction_total_variation": 2.8868818769097683e-05
  },
  "structural_keys_ignoring_duration_airline": {
    "groups": 14905,
    "repeated_groups": 655,
    "repeated_rows": 1339,
    "redundant_rows": 684,
    "conflicting_groups": 91,
    "conflicting_rows": 188,
    "observed_within_key_sse": 3016457197.8333335,
    "observed_within_key_sse_fraction_total_variation": 0.00012639616344499585
  }
}
```

The within-key SSE fraction is a finite-sample observed ambiguity measure, **not a population irreducible-noise bound**. Grouping after arbitrary duration imputation would manufacture false exact matches. Filling deterministic airline redundancy can nevertheless identify semantic duplicates that raw missingness distinguishes.

## Duration reconstruction (feature-only)

Observed duration is masked in five group folds; all identical raw feature rows are blocked together. Fit-only medians use no `price`. This diagnostic measures reconstruction of observed values, not error on genuinely missing values (which could follow a different missingness mechanism), and does not establish original-price R² improvement.

| method | flight_class | n | mae | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| baseline | Business | 5087 | 0.6342 | 2.2806 | -0.3404 |
| baseline | Economy | 7407 | 0.1523 | 1.0344 | -0.0364 |
| departure_arrival | Business | 5087 | 0.4739 | 1.9805 | -0.3004 |
| departure_arrival | Economy | 7407 | 0.1364 | 0.9805 | -0.0335 |
| time_prefix_fallback | Business | 5087 | 0.4708 | 1.9776 | -0.2989 |
| time_prefix_fallback | Economy | 7407 | 0.1298 | 0.9582 | -0.0391 |

| keys | groups_ge3 | rows | weighted_sd | weighted_range | multiple_durations_groups |
| --- | --- | --- | --- | --- | --- |
| flight+route+stops | 1076 | 11682 | 1.0124 | 0.5959 | 280 |
| flight+route+stops+departure_time | 1083 | 11650 | 0.9982 | 0.5668 | 268 |
| flight+route+stops+arrival_time | 1084 | 11540 | 0.4469 | 0.1176 | 219 |
| flight+route+stops+departure_time+arrival_time | 1089 | 11522 | 0.4009 | 0.1073 | 208 |

Departure/arrival-aware keys distinguish itinerary variants that share the first flight identifier, route and stops. The more detailed hierarchy is a candidate only when masked accuracy improves consistently, and must be fitted fold-locally during model evaluation. No inference about Kaggle test missingness is made.

## Heterogeneity and nonlinear effects

| flight_prefix | flight_class | n | price_duration_pearson | price_duration_spearman | days_price_pearson | price_cv |
| --- | --- | --- | --- | --- | --- | --- |
| 6E | Business | 1352 | 0.7117 | 0.7732 | -0.0975 | 0.5219 |
| 6E | Economy | 3797 | 0.0795 | 0.3397 | 0.0288 | 0.3130 |
| AI | Business | 2453 | 0.6155 | 0.6319 | -0.0486 | 0.4073 |
| AI | Economy | 2257 | 0.3609 | 0.4165 | 0.0220 | 0.3422 |
| EK | Business | 330 | -0.0269 | -0.2414 | -0.0191 | 0.1669 |
| EK | Economy | 35 | -0.0063 | -0.2397 | -0.1768 | 0.1979 |
| EY | Business | 290 | -0.3054 | -0.3818 | -0.1140 | 0.3732 |
| EY | Economy | 37 | 0.0714 | 0.2363 | -0.4979 | 0.2378 |
| GF | Business | 24 | -0.1860 | -0.3599 | 0.0925 | 0.0498 |
| IX | Business | 10 | 0.0833 | 0.0800 | 0.1697 | 0.0868 |
| IX | Economy | 405 | 0.2509 | 0.6880 | 0.1212 | 0.3639 |
| MH | Business | 79 | -0.1654 | 0.0199 | 0.1154 | 0.1657 |
| QP | Business | 21 | 0.4104 | -0.0066 | -0.1100 | 0.3530 |
| QP | Economy | 668 | 0.7696 | 0.6197 | 0.1194 | 0.4136 |
| QR | Business | 49 | 0.7637 | 0.3875 | 0.0333 | 0.1780 |
| SG | Economy | 128 | 0.1887 | 0.4071 | -0.4576 | 0.5034 |
| TG | Business | 168 | 0.1509 | 0.3306 | 0.1133 | 0.1539 |
| TG | Economy | 14 | 0.0084 | -0.0338 | -0.4608 | 0.1767 |
| UL | Business | 276 | 0.0905 | 0.1733 | -0.2177 | 0.2912 |
| UL | Economy | 57 | -0.1185 | -0.0221 | -0.4167 | 0.3226 |
| WY | Business | 13 | 0.3564 | 0.4908 | 0.3699 | 0.1277 |

Global duration/price association is strongly confounded with class and prefix. Within Business, domestic-prefix associations differ from EK/EY/TG/UL; a global monotonic duration constraint is unsupported. Preserve raw observed duration and prefix/class interactions; broad numeric polynomials previously failed and should not be repeated.

| flight_prefix | flight_class | n | mean | median | std | min | max | cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SQ | Business | 8 | 213,387.8750 | 205,644.5000 | 33,878.0862 | 155257 | 269912 | 0.1588 |
| GF | Business | 31 | 184,404.5484 | 180,949.0000 | 9,190.0482 | 174594 | 220816 | 0.0498 |
| QR | Business | 60 | 183,147.2833 | 169,353.0000 | 32,605.5004 | 150216 | 245180 | 0.1780 |
| EY | Business | 362 | 152,604.4945 | 147,049.0000 | 56,954.4179 | 61796 | 323177 | 0.3732 |
| EK | Business | 404 | 137,736.0446 | 130,960.0000 | 22,984.4310 | 93120 | 289045 | 0.1669 |
| MH | Business | 94 | 129,020.8936 | 132,072.5000 | 21,381.5382 | 82028 | 165834 | 0.1657 |
| KU | Business | 5 | 121,659.8000 | 121,309.0000 | 784.4126 | 121309 | 123063 | 0.0064 |
| FZ | Business | 11 | 108,003.1818 | 111,529.0000 | 25,803.1615 | 65486 | 147716 | 0.2389 |
| TG | Business | 213 | 107,184.4038 | 102,364.0000 | 16,494.9844 | 65004 | 149753 | 0.1539 |
| QP | Business | 30 | 95,030.9000 | 79,855.0000 | 33,541.8916 | 63140 | 164930 | 0.3530 |
| WY | Business | 24 | 91,399.2500 | 97,078.5000 | 11,670.5350 | 57331 | 106470 | 0.1277 |
| UL | Business | 352 | 80,495.2614 | 71,781.0000 | 23,440.7551 | 49341 | 213680 | 0.2912 |
| RA | Business | 3 | 64,085.3333 | 65,005.0000 | 1,592.9094 | 62246 | 65005 | 0.0249 |
| MH | Economy | 1 | 54,314.0000 | 54,314.0000 |  | 54314 | 54314 |  |
| AI | Business | 3091 | 40,661.1391 | 37,518.0000 | 16,561.3092 | 22562 | 233740 | 0.4073 |
| EK | Economy | 39 | 39,517.2051 | 38,080.0000 | 7,819.8092 | 30145 | 67658 | 0.1979 |
| TG | Economy | 15 | 35,622.0667 | 33,319.0000 | 6,296.0609 | 28537 | 56462 | 0.1767 |
| 6E | Business | 1681 | 34,469.4057 | 30,055.0000 | 17,989.0597 | 15544 | 155467 | 0.5219 |
| EY | Economy | 45 | 34,172.4222 | 32,495.0000 | 8,125.9350 | 20615 | 63720 | 0.2378 |
| IX | Business | 11 | 32,250.9091 | 32,689.0000 | 2,800.9210 | 25886 | 35146 | 0.0868 |
| UL | Economy | 70 | 29,407.9000 | 27,244.5000 | 9,486.8859 | 20212 | 74691 | 0.3226 |
| KU | Economy | 2 | 26,018.0000 | 26,018.0000 | 3,712.3106 | 23393 | 28643 | 0.1427 |
| WY | Economy | 5 | 24,747.4000 | 24,701.0000 | 2,833.0741 | 21937 | 27933 | 0.1145 |
| SG | Economy | 147 | 11,045.8503 | 7,333.0000 | 5,559.9640 | 6668 | 23035 | 0.5034 |
| AI | Economy | 2846 | 8,753.1100 | 8,026.0000 | 2,995.1185 | 4165 | 54494 | 0.3422 |
| QP | Economy | 830 | 8,589.3783 | 7,549.0000 | 3,552.7353 | 4778 | 36554 | 0.4136 |
| 6E | Economy | 4701 | 8,077.5273 | 7,435.0000 | 2,528.4403 | 3801 | 33246 | 0.3130 |
| S5 | Economy | 3 | 7,739.3333 | 7,099.0000 | 1,109.0899 | 7099 | 9020 | 0.1433 |
| IX | Economy | 503 | 7,222.0557 | 6,712.0000 | 2,628.0855 | 3874 | 20440 | 0.3639 |
| 9I | Economy | 2 | 5,084.0000 | 5,084.0000 | 240.4163 | 4914 | 5254 | 0.0473 |

The top-1% price cohort contains 156 rows, all Business. Its prefix distribution is `{'EY': 102, 'EK': 18, 'QR': 18, 'SQ': 7, 'GF': 6, 'UL': 3, 'AI': 2}`. Sparse high-price prefix groups mean a few expensive examples can change pooled R² markedly. Rare-category collapse should therefore be audited: an uncommon airline may describe expensive structural conditions, not arbitrary noise. Avoid direct target-history encoding and price weighting already rejected in prior work.

| flight_class | days_bin | n | mean | median | std | min | max | cv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Business | (-0.001, 3.0] | 302 | 71,111.6093 | 48,562.0000 | 52,102.0088 | 21345 | 313132 | 0.7327 |
| Business | (14.0, 21.0] | 1031 | 61,497.3104 | 37,737.0000 | 48,811.8726 | 15544 | 300148 | 0.7937 |
| Business | (21.0, 30.0] | 1345 | 60,500.9985 | 38,426.0000 | 46,191.9208 | 18085 | 289045 | 0.7635 |
| Business | (3.0, 7.0] | 614 | 64,972.0684 | 41,124.0000 | 53,121.8666 | 22562 | 323177 | 0.8176 |
| Business | (30.0, 40.0] | 1501 | 58,357.8068 | 38,426.0000 | 44,519.5329 | 15581 | 317854 | 0.7629 |
| Business | (40.0, 50.0] | 371 | 40,729.6846 | 31,688.0000 | 22,111.5489 | 19918 | 146624 | 0.5429 |
| Business | (50.0, 60.0] | 144 | 52,195.7222 | 38,546.0000 | 38,344.9826 | 22792 | 221059 | 0.7346 |
| Business | (7.0, 14.0] | 1072 | 62,723.5672 | 43,234.0000 | 46,300.8987 | 22562 | 308302 | 0.7382 |
| Economy | (-0.001, 3.0] | 655 | 10,793.8031 | 9,099.0000 | 7,417.0171 | 4129 | 74691 | 0.6872 |
| Economy | (14.0, 21.0] | 1540 | 8,470.5857 | 7,871.5000 | 3,842.8349 | 3801 | 38080 | 0.4537 |
| Economy | (21.0, 30.0] | 1860 | 8,273.3204 | 7,277.0000 | 4,132.9966 | 3801 | 54423 | 0.4996 |
| Economy | (3.0, 7.0] | 758 | 8,203.1873 | 7,463.0000 | 3,616.6144 | 3895 | 56638 | 0.4409 |
| Economy | (30.0, 40.0] | 2018 | 9,493.9341 | 7,587.0000 | 4,999.3690 | 3801 | 52292 | 0.5266 |
| Economy | (40.0, 50.0] | 562 | 7,846.7367 | 7,070.0000 | 2,448.6275 | 5985 | 30145 | 0.3121 |
| Economy | (50.0, 60.0] | 101 | 11,350.9802 | 10,036.0000 | 5,074.4354 | 5758 | 36419 | 0.4470 |
| Economy | (7.0, 14.0] | 1715 | 8,611.4303 | 7,716.0000 | 4,728.9691 | 3874 | 48806 | 0.5492 |

Daily averages mix prefix/route composition and ticket selection, and do not justify causal monotonicity. `days_within_cell_patterns.csv` provides descriptive demeaning by prefix/class/route/stops; these full-training label summaries must never be used as model features. Existing trees already capture thresholds; additional lead-time transforms require isolated ablations rather than repetition of the failed feature bundle.

## Recommendations and limits

1. Compare the exact original XGB850 control with a single departure/arrival-aware duration hierarchy if masked reconstruction supports it. Preserve missingness indicators and raw duration. Add reliability/count features only as a separate justified ablation; no arbitrary target-conditioned imputation.
2. Audit categorical representation and sparsity: keep interpretable prefix/class information and consider a bounded categorical model only if it addresses OHE rare-flight collapse. Avoid conflating a new model family, target transform and feature bundle.
3. Use raw identical-row grouping as primary historical comparison and canonical airline-filled grouping as a sensitivity check. Structural keys that omit duration intentionally make a harder task and do not imply identical observations.
4. Do not remove extreme observations or guarantee gains. No values violate basic positivity checks; high prices occur in recognizable but sparse groups. Validate original-price errors by prefix/class and top-tail contribution.
5. A fresh fold seed does not create wholly untouched labels after broad historic/EDA use. Use nested inner selection and predeclared outer comparisons, label uncertainty honestly, and keep Kaggle inference until all choices are frozen.

Generated by `training_eda.py`; CSVs retain exact numerical diagnostics and four PNG figures visualize the evidence.
