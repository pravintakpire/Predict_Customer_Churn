# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source /Users/pravintakpire/datascience/datascient/bin/activate
```

Jupyter notebooks use the pre-configured kernel named `datascient`.

## Common Commands

```bash
python src/explore.py                  # data sanity check
python src/train.py                    # baseline LightGBM → submission_lgbm_baseline.csv
python src/tune.py                     # LightGBM Optuna tuning (15 trials, prints params)
python src/train_tuned.py              # LightGBM with pre-tuned params → submission_lgbm_tuned.csv
python src/xgboost_multiseed_fe.py     # XGBoost 50-model ensemble → submission_xgboost_multiseed_fe.csv
python src/tune_xgboost.py             # XGBoost Optuna tuning with xgb.cv() → xgboost_tuning_results.csv
```

## Architecture & Data Flow

**Data**: `data/train.csv` (80.5 MB) and `data/test.csv` (33.8 MB), Kaggle competition `playground-series-s6e3`. Target column is `Churn` (Yes/No → 1/0).

**Preprocessing pattern** (consistent across all scripts/notebooks):
1. Combine train + test for joint label encoding (avoids category mismatch)
2. Fill numeric NaN with median, categorical NaN with `'Missing'`
3. Split back before modeling

**Feature engineering** (all advanced notebooks):
- Row stats on `[tenure, MonthlyCharges, TotalCharges]`: `num_sum`, `num_mean`, `num_std`, `num_max`, `num_min`
- Domain features: `Average_Monthly_Actual = TotalCharges/(tenure+1e-5)`, `Monthly_diff`, `Monthly_ratio`

**Outputs**: All submission CSVs written to project root. Trained models not persisted.

---

## Notebook Progression

| Notebook | Model | Key Technique | Status |
|----------|-------|---------------|--------|
| `00_general_eda.ipynb` | — | General EDA + `DataTransformer` class (fit on train, apply to test) | Reusable |
| `01_eda.ipynb` | — | Competition-specific EDA | Reference |
| `02_kaggle_gpu_pipeline.ipynb` | LightGBM | Optuna 20 trials, 5-fold CV | Old |
| `03_kaggle_xgboost_gpu_pipeline.ipynb` | XGBoost | Optuna 20 trials, 5-fold CV | Old |
| `04_kaggle_xgboost_multiseed_100t.ipynb` | XGBoost | 100 trials, `xgb.cv()` in Optuna, 50 models | Old (no FE) |
| `05_kaggle_lightgbm_multiseed_100t.ipynb` | LightGBM | 100 trials, `lgb.cv()` in Optuna, 50 models | Old (no FE) |
| `06_xgboost_tuned_multiseed_fe.ipynb` | XGBoost | 100 trials, `xgb.cv()`, FE, tqdm, GPU detect | **Best XGBoost** |
| `07_catboost_tuned_multiseed_fe.ipynb` | CatBoost | Native categoricals, Optuna, 50 models | Slower, lower score |
| `08_blend_submissions.ipynb` | Ensemble | Weighted average + rank averaging of CSVs | **Current best** |
| `09_lightgbm_tuned_multiseed_fe.ipynb` | LightGBM | 100 trials, `lgb.cv()`, FE, tqdm, GPU detect | **Use for blending** |
| `10_xgboost_fulldata_scaledpos.ipynb` | XGBoost | `scale_pos_weight`, `grow_policy`, full-data retrain trick | **Latest** |

---

## Competition Score History

| Submission | Public Score | Notes |
|------------|-------------|-------|
| `submission_lgbm_baseline.csv` | — | Baseline, no tuning |
| `submission_xgboost_multiseed_fe.csv` | **0.91417** | 50-model ensemble, FE |
| `submission_xgboost_multiseed_fe_100_6.csv` | 0.91415 | 100 trials, 6-fold CV |
| `submission_catboost_tuned_multiseed_fe.csv` | 0.91340 | CatBoost — worse, abandoned |
| `submission_blend_custom.csv` | **0.91433** | XGB×2 + LGB×1.5 + XGB-100n×1 |
| `submission_blend_custom_rank.csv` | **0.91433** | Same weights, rank-averaged |
| `submission_blend_equal.csv` | 0.91427 | All 5 models equal weight |
| `submission_blend_score_weighted.csv` | 0.91406 | LB-score weighted (CatBoost dragged it down) |

**Current best: 0.91433** (blend_custom / blend_custom_rank)

---

## Key Learnings & Decisions

**What works:**
- `xgb.cv()` / `lgb.cv()` *inside* each Optuna trial — far more robust than single 80/20 split tuning
- Multi-seed ensemble (5 seeds × 10 folds = 50 models) reduces variance significantly
- Blending XGBoost + LightGBM gives +0.0002 over best single model
- Rank averaging tied with direct blend — probability scales are already aligned
- Feature engineering (8 features on 3 numeric cols) consistently helps

**What doesn't work:**
- CatBoost: slower and lower score (0.91340) — low-cardinality categoricals give it no edge here
- Score-weighted blending when a weak model (CatBoost) is included — it drags the blend down
- More tuning trials beyond 50–100: XGBoost plateaued at 0.91417 with both 50 and 100 trials
- `colsample_bylevel` in CatBoost on GPU — crashes with `rsm not supported for non-pairwise on GPU`

**Not yet tried (next steps):**
- `scale_pos_weight` tuned via Optuna (notebook 10 — not yet submitted)
- `grow_policy='lossguide'` + `max_leaves` in XGBoost (notebook 10)
- Full data retrain trick: `full_rounds = best_round × K/(K-1)` (notebook 10)
- LightGBM with feature engineering (notebook 09 — not yet submitted)
- Blending notebook 09 (LGB+FE) output into notebook 08

---

## Blending Setup (`notebooks/08_blend_submissions.ipynb`)

The blend notebook reads submission CSVs from the project root. To add a new model:
```python
# In the SUBMISSIONS list in cell-3:
{'file': 'your_submission.csv', 'label': 'MyModel', 'public_score': 0.914xx},
```

Custom weights currently used (best blend):
```python
CUSTOM_WEIGHTS = {
    'XGB-50t-FE':     2.0,
    'XGB-100t-6fold': 2.0,
    'LGB-100n':       1.5,
    'XGB-100n':       1.0,
    'CatBoost':       0.0,   # excluded — lower score
}
```

**Kaggle submission commands (latest):**
```bash
kaggle competitions submit -c playground-series-s6e3 -f submission_blend_custom_rank.csv -m "Blend: Custom rank-avg — XGB-50t×2 + XGB-100t×2 + LGB-100n×1.5 + XGB-100n×1"
kaggle competitions submit -c playground-series-s6e3 -f submission_xgb_scaledpos_ensemble.csv -m "XGB scaledpos+lossguide ensemble 50 models"
kaggle competitions submit -c playground-series-s6e3 -f submission_xgb_scaledpos_fulldata.csv -m "XGB scaledpos fulldata retrain"
kaggle competitions submit -c playground-series-s6e3 -f submission_xgb_scaledpos_blend.csv -m "XGB scaledpos ensemble+fulldata 50/50 blend"
```

---

## CatBoost GPU Gotcha

`colsample_bylevel` maps to CatBoost's `rsm` parameter which is **not supported on GPU** for non-pairwise loss functions. Always gate it:
```python
if not USE_GPU:
    params['colsample_bylevel'] = trial.suggest_float(...)
```
