#!/usr/bin/env bash
set -euo pipefail
cd /workspace/airfare_fresh
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/airfare-mpl
python -m venv --system-site-packages /workspace/airfare_research/.venv
/workspace/airfare_research/.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
/workspace/airfare_research/.venv/bin/python -c 'import numpy,pandas,scipy,sklearn,xgboost,lightgbm,catboost,optuna; print("Independent airfare research dependencies verified")'
