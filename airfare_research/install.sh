#!/usr/bin/env bash
set -euo pipefail
cd /workspace/airfare_research
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
.venv/bin/python -c 'import numpy, pandas, scipy, sklearn, xgboost, lightgbm, catboost; print("Airfare research imports verified")'
