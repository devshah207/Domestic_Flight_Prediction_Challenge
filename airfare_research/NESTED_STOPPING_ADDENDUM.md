# Adaptive addition, declared before its scores

Native850 increased mean train R² to.9810 and decreased outer OOF R² to.92799
versus sparse OHE control.92981. This motivates one finite stopping procedure
within the already registered native representation family, without a depth or
penalty search. This adaptation used previous outer diagnostics and is exploratory;
primary outer folds are not a pristine independent selection test.

Procedure frozen now: each outer training partition creates five raw-duplicate-
blocked decile-stratified inner folds seed=outer_seed+1000+outer_fold; the first two
inner folds each select best_iteration+1 on original-price-equivalent squared
error, cap1400 patience75. Fit the complete feature hierarchy and category levels
on inner fitting rows separately. Median of the two stops, floor100, refits all
outer training and scores outer fold once. Keep all other native parameters
unchanged. One predeclared candidate, no additional variants. Compare to fixed850
control, apply original material-effect/fold/interval gate, include4-procedure
multiplicity sensitivity, and confirm any selected candidate unchanged on seed
20261016. Do not claim repeated folds supply new independent labels.

Full-data rounds, only if this procedure qualifies: evaluate stopping at two
predeclared training fractions .64 and .80 (five fold-pure fits each), report
sample-size/round variability, and use a conservative capped log-linear mapping
to100% only if stable; otherwise retain the predeclared fixed850 winner or baseline.
No final-round selection from Kaggle features or predictions.
