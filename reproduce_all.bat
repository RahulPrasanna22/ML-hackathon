@echo off
REM Full pipeline: EDA -> dataset -> models+CV -> Bayesian opt -> final model -> decoder tuning -> evaluation -> plots -> tests
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
%PY% scripts\01_explore.py || exit /b 1
%PY% scripts\02_build_dataset.py || exit /b 1
%PY% scripts\03_train_models.py || exit /b 1
%PY% scripts\04_bayes_opt.py --model xgboost --trials 40 --timeout 3000 || exit /b 1
%PY% scripts\05_finalize_model.py --model xgboost || exit /b 1
%PY% scripts\06_tune_decoder.py --trials 30 --n-trips 24 || exit /b 1
%PY% scripts\07_evaluate_routes.py --maps || exit /b 1
%PY% scripts\08_make_plots.py || exit /b 1
%PY% scripts\09_benchmark.py || exit /b 1
%PY% -m pytest -q tests
