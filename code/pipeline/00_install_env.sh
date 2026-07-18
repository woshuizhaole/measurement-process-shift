#!/usr/bin/env bash
# Stage 0: 安装项目本地环境（不污染全局）。必须在 unset PYTHONPATH 下运行。
set -uo pipefail
P=.
unset PYTHONPATH; export PYTHONNOUSERSITE=1
PIP="$P/env/conda/bin/pip"
run(){ echo "### $*"; "$@" || { echo "!!! FAILED: $*"; return 1; }; }

run $PIP install --no-input -c $P/env/constraints.txt --upgrade pip setuptools wheel
run $PIP install --no-input -c $P/env/constraints.txt --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121
run $PIP install --no-input -c $P/env/constraints.txt "numpy==1.26.4" "scipy==1.13.1" "pandas==2.2.3" "scikit-learn==1.5.2" \
    "polars==1.12.0" "numba==0.60.0" "llvmlite==0.43.0"
# 偏离: statsmodels 0.14.4 -> 0.14.6 (dcurves 1.1.7 要求 >=0.14.5)；pyarrow<18 (mlflow 2.17.2 要求)
run $PIP install --no-input -c $P/env/constraints.txt "statsmodels==0.14.6" "pyarrow==17.0.0"
run $PIP install --no-input -c $P/env/constraints.txt "shap==0.46.0" "lightgbm==4.5.0" "xgboost==2.1.4" \
    "hydra-core==1.3.2" "mlflow==2.17.2" "autorank" \
    "matplotlib" "seaborn" "tqdm" "joblib" "pyyaml" "tabulate"
run $PIP install --no-input -c $P/env/constraints.txt "dcurves==1.1.7"
run $PIP install --no-input -c $P/env/constraints.txt "pypots==1.5" "pygrinder==0.7" "benchpots==1" "tsdb==0.8"
run $PIP install --no-input -c $P/env/constraints.txt "densratio" "dcor" "miceforest"
$PIP install --no-input -c $P/env/constraints.txt "abstention" || echo "WARN abstention pypi failed -> clone github"
echo "=== pip check ==="; $PIP check || true
echo "=== freeze ==="; $PIP freeze > $P/env/pip_freeze.txt; wc -l $P/env/pip_freeze.txt
echo "INSTALL_DONE"
