# High-round Optuna boosting results

All scores below are training-only estimates. No new Kaggle R² or rank is known. All successful candidates are delivered under the user’s explicit request for multiple aggressive exploratory submissions; weaker results are not concealed.

| model | r2 | delta_vs_business | mean_train_r2 | mean_gap | worst_fold_r2 | folds_won_vs_business | final_actual_rounds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control_previous_equal64 | 0.928594 | 0.001741 | — | — | 0.919705 | 4 | — |
| equal_dart3 | 0.927333 | 0.000480 | 0.968806 | 0.041439 | 0.920253 | 3 | — |
| equal_main3 | 0.927105 | 0.000253 | 0.969857 | 0.042712 | 0.918579 | 2 | — |
| control_business_augmented | 0.926853 | 0.000000 | 0.965137 | 0.038239 | 0.916187 | 0 | — |
| control_global_xgb | 0.925416 | -0.001437 | 0.967967 | 0.042506 | 0.916717 | 1 | — |
| xgb | 0.924665 | -0.002188 | 0.971941 | 0.047242 | 0.916971 | 1 | 1,400.000000 |
| lgb_gbdt | 0.923440 | -0.003413 | 0.967860 | 0.044385 | 0.912168 | 1 | 1,400.000000 |
| hist_gradient | 0.923166 | -0.003687 | 0.966341 | 0.043152 | 0.915009 | 1 | 800.000000 |
| cat_symmetric | 0.922904 | -0.003949 | 0.962873 | 0.039935 | 0.914577 | 0 | 1,400.000000 |
| classic_gradient | 0.922722 | -0.004131 | 0.958221 | 0.035480 | 0.915648 | 0 | 1,100.000000 |
| cat_leafwise | 0.922688 | -0.004165 | 0.955016 | 0.032297 | 0.911799 | 2 | 1,400.000000 |
| lgb_dart | 0.922404 | -0.004449 | 0.962694 | 0.040262 | 0.912946 | 1 | 1,400.000000 |
| adaboost | 0.914292 | -0.012561 | 0.943767 | 0.029444 | 0.905124 | 0 | 1,100.000000 |

## Tail and subgroup diagnostics

| model | top1_rmse | top1_bias | top1_sse_share | business_sse_share |
| --- | --- | --- | --- | --- |
| xgb | 63,617.25251 | -48,008.80223 | 0.35117 | 0.97078 |
| lgb_gbdt | 65,393.43079 | -50,636.29091 | 0.36511 | 0.96944 |
| lgb_dart | 70,021.04301 | -57,713.61656 | 0.41303 | 0.97612 |
| cat_symmetric | 64,158.96690 | -49,415.58805 | 0.34902 | 0.97375 |
| cat_leafwise | 65,027.82795 | -50,790.32711 | 0.35753 | 0.97388 |
| hist_gradient | 66,149.09173 | -51,369.57287 | 0.37227 | 0.97226 |
| classic_gradient | 66,488.40833 | -52,881.59577 | 0.37394 | 0.97560 |
| adaboost | 65,902.01769 | -53,625.64545 | 0.33124 | 0.97595 |
| control_global_xgb | 62,818.57206 | -48,041.49131 | 0.34585 | 0.96748 |
| control_business_augmented | 63,698.31755 | -50,267.06739 | 0.36259 | 0.96684 |
| control_previous_equal64 | 63,077.25579 | -49,500.50147 | 0.36423 | 0.97349 |
| equal_main3 | 63,783.61800 | -49,353.56040 | 0.36482 | 0.97559 |
| equal_dart3 | 65,097.29819 | -51,712.66895 | 0.38120 | 0.97770 |

## Interpretation and limits

Each outer fold tunes only inside its fitting partition, then refits at a fixed selected high count before scoring the outer validation fold. Outer folds are shared with previous research and are not a fresh untouched test. Each scope uses only one inner 80/20 split to contain runtime; hyperparameters can be unstable. Selection among eight families also creates optimism. Bootstrap intervals condition on saved OOF predictions and omit retraining, tuning and accumulated research uncertainty.

The previous equal-blend64 OOF is included as an additional descriptive control on verified identical rows/folds, so a gain over the Business control is not presented as beating every earlier CV result. Its composite training predictions were not retained in that phase; its training gap is left missing rather than estimated from component gaps. No new fitting or changes to any submitted blend weights are introduced by this comparison.

The explicit 800-round floor may be suboptimal; these are requested high-capacity candidates, not claims of optimal count. Actual rounds are reported separately because AdaBoost and some libraries can terminate before a requested cap. A larger training gap is a warning, not proof of its cause. Tree count alone does not establish overfitting or improve the public leaderboard.

A scheduling race caused a second runner to enter the existing LightGBM outer0 scope while original trial2 was suspended. That exact trial was recovered using identical parameters/inner rows, without a new suggestion. The original duplicate later fails its second completion update for the already-finished trial, preventing further duplicate suggestions. There are 336 distinct Optuna trials and 384 canonical research fits, plus one duplicate fit attempt (385 total). This one scope includes asynchronous/restarted TPE ordering; saved trials provide the exact parameters. Other scopes follow the registered seeded procedure. See SCHEDULING_RECOVERY.json and logs.

Exact final dictionaries, requested and actual counts, feature schemas and resolved library parameters are in MODEL_CONFIGURATIONS.json. Trials, selections, fold indices, errors, confidence intervals and fit accounting are retained. Both blends have registered fixed equal weights; none were fit to outer or public scores. Existing models and submissions remain unchanged.

## Reproduction

Use the pinned ../airfare_fresh/requirements.txt. From the repository root set OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1, then run python airfare_boosting_sweep/study.py --stage register --train TRAIN.csv; python airfare_boosting_sweep/study.py --stage run --train TRAIN.csv; python airfare_boosting_sweep/analyze.py --train TRAIN.csv. Run in a fresh copy without generated study outputs to reproduce fitting. Registration prevents silent code/data/version changes. Inference has a one-use marker and never permits research after test access.
