# Phase 4 targeted EDA (training observations only)

These statistics are computed exclusively from `train(1).csv`. They justify the predeclared engineered features but do not demonstrate predictive benefit. That is evaluated with leakage-safe CV.

## Price patterns by flight_class
| flight_class   |   count |   median |     mean |
|:---------------|--------:|---------:|---------:|
| Economy        |    9209 |     7587 |  8817.97 |
| Business       |    6380 |    38546 | 60126.6  |

## Price patterns by stops
| stops       |   count |   median |     mean |
|:------------|--------:|---------:|---------:|
| zero        |   10965 |     8883 |  15507.6 |
| one         |    4558 |    40528 |  62782.2 |
| two_or_more |      66 |   125611 | 130463   |

## Price patterns by departure_time
| departure_time   |   count |   median |    mean |
|:-----------------|--------:|---------:|--------:|
| Evening          |    3304 |   9450   | 20972.6 |
| Morning          |    3296 |  10864.5 | 33654.1 |
| Night            |    2832 |  11000   | 33497.8 |
| Early_Morning    |    2680 |   9943   | 30279   |
| Afternoon        |    2462 |   9802   | 22147.1 |
| Late_Night       |    1015 |  33067   | 53256.7 |

## Price patterns by arrival_time
| arrival_time   |   count |   median |    mean |
|:---------------|--------:|---------:|--------:|
| Night          |    3845 |  11466   | 32259.1 |
| Morning        |    3116 |   9883   | 30429.2 |
| Evening        |    2968 |  10045   | 27693.1 |
| Afternoon      |    2860 |   9846   | 25931.5 |
| Late_Night     |    1826 |  17179.5 | 38409.9 |
| Early_Morning  |     974 |   9181   | 19985.1 |

## Price patterns by source_city
| source_city   |   count |   median |    mean |
|:--------------|--------:|---------:|--------:|
| Metropolis    |    6198 |   8976   | 17740   |
| Night City    |    2651 |  10352   | 22942.4 |
| Springfield   |    2051 |  13018   | 39006.5 |
| Hill Valley   |    1600 |  35504.5 | 45538.7 |
| Gotham        |    1586 |  28421   | 45683.6 |
| Los Santos    |    1503 |  21345   | 45722.7 |

## Price patterns by destination_city
| destination_city   |   count |   median |    mean |
|:-------------------|--------:|---------:|--------:|
| Night City         |    4663 |   7364   | 19314.5 |
| Springfield        |    2870 |  10884.5 | 34335.5 |
| Metropolis         |    2759 |   9880   | 18627.3 |
| Hill Valley        |    2126 |  11551   | 34507.8 |
| Los Santos         |    1609 |  24049   | 45580.8 |
| Gotham             |    1562 |  30084   | 50006.7 |

## Booking lead-time × cabin class
|                                                     |   count |   median |     mean |
|:----------------------------------------------------|--------:|---------:|---------:|
| ('Business', Interval(-0.001, 3.0, closed='right')) |     302 |    48562 | 71111.6  |
| ('Business', Interval(3.0, 7.0, closed='right'))    |     614 |    41124 | 64972.1  |
| ('Business', Interval(7.0, 14.0, closed='right'))   |    1072 |    43234 | 62723.6  |
| ('Business', Interval(14.0, 30.0, closed='right'))  |    2376 |    38129 | 60933.3  |
| ('Business', Interval(30.0, 60.0, closed='right'))  |    2016 |    37771 | 54673.6  |
| ('Economy', Interval(-0.001, 3.0, closed='right'))  |     655 |     9099 | 10793.8  |
| ('Economy', Interval(3.0, 7.0, closed='right'))     |     758 |     7463 |  8203.19 |
| ('Economy', Interval(7.0, 14.0, closed='right'))    |    1715 |     7716 |  8611.43 |
| ('Economy', Interval(14.0, 30.0, closed='right'))   |    3400 |     7473 |  8362.67 |
| ('Economy', Interval(30.0, 60.0, closed='right'))   |    2681 |     7435 |  9218.6  |

## Observed trip duration × cabin class (only measured durations)
|                                                     |   count |   median |      mean |
|:----------------------------------------------------|--------:|---------:|----------:|
| ('Business', Interval(-0.001, 2.0, closed='right')) |     313 |  24049   |  25514.6  |
| ('Business', Interval(2.0, 4.0, closed='right'))    |    1956 |  36614   |  37763.7  |
| ('Business', Interval(4.0, 6.0, closed='right'))    |     940 |  29499   |  35704.8  |
| ('Business', Interval(6.0, 10.0, closed='right'))   |     866 |  50167   |  77350    |
| ('Business', Interval(10.0, 15.0, closed='right'))  |     545 | 129288   | 125957    |
| ('Business', Interval(15.0, 50.0, closed='right'))  |     467 | 111720   | 120085    |
| ('Economy', Interval(-0.001, 2.0, closed='right'))  |    1056 |   6476   |   6663.43 |
| ('Economy', Interval(2.0, 4.0, closed='right'))     |    5529 |   8259   |   8731.06 |
| ('Economy', Interval(4.0, 6.0, closed='right'))     |     594 |   7534.5 |   8265.46 |
| ('Economy', Interval(6.0, 10.0, closed='right'))    |     114 |   8109   |  15033.9  |
| ('Economy', Interval(10.0, 15.0, closed='right'))   |      50 |  35429   |  34292.6  |
| ('Economy', Interval(15.0, 50.0, closed='right'))   |      64 |  32562   |  33002.5  |

## Route × cabin class (at least 50 rows per group)
|                                           |   count |   median |     mean |
|:------------------------------------------|--------:|---------:|---------:|
| ('Business', 'Los Santos → Springfield')  |     162 | 130983   | 119248   |
| ('Business', 'Springfield → Los Santos')  |     164 | 130960   | 128192   |
| ('Business', 'Los Santos → Gotham')       |     150 | 129979   | 117602   |
| ('Business', 'Gotham → Springfield')      |     192 | 126629   | 117544   |
| ('Business', 'Springfield → Gotham')      |     190 |  95231   | 106628   |
| ('Business', 'Hill Valley → Los Santos')  |     184 |  88604   |  86254.1 |
| ('Business', 'Hill Valley → Gotham')      |     226 |  69829.5 |  81952.9 |
| ('Business', 'Gotham → Hill Valley')      |     192 |  69773   |  87693.9 |
| ('Business', 'Gotham → Los Santos')       |     216 |  56651.5 |  78249.6 |
| ('Business', 'Hill Valley → Springfield') |     207 |  47884   |  77483.4 |
| ('Business', 'Springfield → Hill Valley') |     165 |  47141   |  72656.1 |
| ('Business', 'Los Santos → Hill Valley')  |     161 |  46175   |  88496.1 |
| ('Business', 'Night City → Hill Valley')  |     245 |  40038   |  53905.5 |
| ('Business', 'Springfield → Metropolis')  |     275 |  38546   |  42121.4 |
| ('Business', 'Metropolis → Springfield')  |     375 |  38426   |  40987.5 |
| ('Business', 'Hill Valley → Night City')  |     133 |  38129   |  53494.6 |
| ('Business', 'Gotham → Metropolis')       |     111 |  37271   |  39178.7 |
| ('Business', 'Hill Valley → Metropolis')  |     163 |  34819   |  47661.1 |
| ('Business', 'Metropolis → Night City')   |    1153 |  33712   |  39100   |
| ('Business', 'Night City → Metropolis')   |     142 |  32859   |  39891   |
| ('Business', 'Metropolis → Los Santos')   |      76 |  32815   |  40992.1 |
| ('Business', 'Metropolis → Hill Valley')  |      67 |  32507   |  46961   |

## Leakage/ambiguity risks
- Repeated predictor-row groups: 310 groups, totaling 624 rows.
- Conflicting prices despite identical predictors: 35 groups.
- Same predictor rows are blocked from crossing five-fold validation boundaries.
- Group-level category/median imputation is fitted only within each training fold; validation prices never create features.

## Predeclared feature hypotheses
- Class-specific duration and days-left interactions because observed class price scales differ dramatically.
- Route × airline, route × departure period, and cabin × stops/departure-time categorical interactions.
- Smooth numeric nonlinearities for duration and days-left, and duration per trip leg.
- Rare flight/route pairs aggregated to avoid thousands of nearly empty indicator columns.

No feature is included solely because it improves the public Kaggle score.
