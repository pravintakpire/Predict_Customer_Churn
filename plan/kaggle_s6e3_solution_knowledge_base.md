# Kaggle PS S6E3 — Predict Customer Churn: Solution Knowledge Base

**Competition**: Playground Series Season 6, Episode 3 — Predict Customer Churn
**URL**: https://www.kaggle.com/competitions/playground-series-s6e3
**Metric**: ROC-AUC (binary classification)
**Source**: Mined from local competition notebooks + web research
**Date compiled**: 2026-03-27

---

## 1. Competition Brief

| Field | Detail |
|-------|--------|
| Task | Binary classification — predict whether a telecom customer will churn (Yes/No) |
| Data source | Synthetically generated from IBM Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) |
| Train size | 594,194 rows × 21 columns |
| Test size | 254,655 rows × 20 columns |
| Original dataset | 7,043 rows (IBM WA_Fn dataset — same schema, provided in competition data) |
| Target column | `Churn` (Yes → 1, No → 0) |
| Class imbalance | 22.5% churn (Yes=133,817, No=460,377) → `scale_pos_weight ≈ 3.44` |
| Evaluation metric | Area Under ROC Curve (AUC) |

### Features (20 input features)

| Feature | Type | Notes |
|---------|------|-------|
| `gender` | Categorical | Male/Female — low signal (< 2pp churn diff) |
| `SeniorCitizen` | Binary int | 0/1 |
| `Partner` | Categorical | Yes/No |
| `Dependents` | Categorical | Yes/No |
| `tenure` | Numeric | Months with company [1–72], mean=36.6 |
| `PhoneService` | Categorical | Yes/No — low signal (< 2pp) |
| `MultipleLines` | Categorical | Yes/No/No phone service |
| `InternetService` | Categorical | DSL (10.3% churn) / Fiber optic (41.5% churn!) / No (1.4%) |
| `OnlineSecurity` | Categorical | Yes/No/No internet |
| `OnlineBackup` | Categorical | Yes/No/No internet |
| `DeviceProtection` | Categorical | Yes/No/No internet |
| `TechSupport` | Categorical | Yes/No/No internet |
| `StreamingTV` | Categorical | Yes/No/No internet |
| `StreamingMovies` | Categorical | Yes/No/No internet |
| `Contract` | Categorical | Month-to-month (42.1% churn!) / One year (5.8%) / Two year (1.0%) |
| `PaperlessBilling` | Categorical | Yes/No |
| `PaymentMethod` | Categorical | Electronic check (48.9% churn!) / Bank transfer / Credit card / Mailed check (~7.7%) |
| `MonthlyCharges` | Numeric | [18.25–118.75], mean=65.9 |
| `TotalCharges` | Numeric | [18.8–8684.8], mean=2494.4 |
| `id` | Integer | Row identifier (not a feature) |

### Key EDA Findings

| Signal | Churn Rate | Lift vs Overall |
|--------|-----------|-----------------|
| Fiber optic internet | 41.5% | 1.84x |
| Electronic check payment | 48.9% | 2.17x |
| Month-to-month contract | 42.1% | 1.87x |
| Fiber optic + 0 security services | ~55%+ | 2.40x |
| Electronic check + month-to-month | ~55%+ | 2.02x |
| New customer (tenure ≤ 12) + MTM | high | 1.94x |
| Senior citizen, no partner/dependents | elevated | notable |
| Two-year contract | 1.0% | 0.04x (strong retention signal) |
| `gender` | negligible | < 2pp spread — drop recommended |
| `PhoneService` | negligible | < 2pp spread — drop recommended |

---

## 2. Original Summaries of Top Solutions

> **Note**: Kaggle's competition pages require JavaScript rendering; direct scraping of the official leaderboard was not feasible. The following is synthesized from: (1) all notebooks developed locally for this competition, (2) analysis of the tuning logs and submission history, and (3) cross-referencing against patterns from adjacent Playground Series competitions (S5E11 1st place, S6E2 1st place, general 2025 Kaggle winner patterns). Current local best score is **0.91433** (public LB).

### Best submission scores achieved (local history)

| Submission | Public AUC | Method |
|-----------|-----------|--------|
| `submission_lgbm_baseline.csv` | ~0.912 | Baseline LightGBM, no tuning |
| `submission_xgboost_multiseed_fe.csv` | **0.91417** | XGBoost 50-model ensemble + FE |
| `submission_xgboost_multiseed_fe_100_6.csv` | 0.91415 | 100 trials, 6-fold CV |
| `submission_catboost_tuned_multiseed_fe.csv` | 0.91340 | CatBoost — abandoned |
| `submission_blend_custom.csv` | **0.91433** | XGB×2 + LGB×1.5 + XGB-100n×1 |
| `submission_blend_custom_rank.csv` | **0.91433** | Same weights, rank-averaged |
| `submission_blend_equal.csv` | 0.91427 | All 5 models equal weight |
| `submission_xgb_scaledpos_blend.csv` | ~0.914 | scale_pos_weight + lossguide + fulldata blend |

---

## 3. Detailed Technical Analysis of Top Techniques

### Technique Tier 1: Multi-Seed Cross-Validation Ensemble (Core Pattern)

**Philosophy**: Train N_seeds × N_folds models, average all predictions. Single best model ≈ 0.912; 50-model ensemble ≈ 0.914. Variance reduction is the primary gain.

**Implementation**:
```python
SEEDS    = [42, 2024, 777, 1337, 999]    # 5 seeds
N_SPLITS = 10                             # 10-fold CV
TOTAL_MODELS = 50                         # 5 × 10

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = xgb.train(params, dtrain, ...)
    test_preds_sum += model.predict(dtest)

final_preds = test_preds_sum / TOTAL_MODELS
```

**Key rule**: All seeds must use StratifiedKFold (not random split) to maintain class balance across folds.

---

### Technique Tier 2: Optuna Hyperparameter Tuning with xgb.cv() / lgb.cv()

**Philosophy**: Run full K-fold CV inside each Optuna trial. Far more robust than single 80/20 split. Best AUC CV achieved: **0.9167** (EDA FE + XGB).

**XGBoost Optuna search space**:
```python
def objective(trial):
    params = {
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "max_leaves":        trial.suggest_int("max_leaves", 15, 127),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight":  trial.suggest_float("min_child_weight", 1, 20),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "gamma":             trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # nb10+ additions:
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", spw * 0.5, spw * 1.5),
        "grow_policy":       trial.suggest_categorical("grow_policy", ["depthwise","lossguide"]),
        "max_leaves":        trial.suggest_int("max_leaves", 15, 255),  # if lossguide
        "max_bin":           trial.suggest_int("max_bin", 128, 512),
    }
    cv = xgb.cv(params, dtrain_full, num_boost_round=1000, nfold=5, ...)
    return cv["test-auc-mean"].max()
```

**Best params found (EDA FE notebook, trial #19)**:
```python
{
    "learning_rate":    0.060562,
    "max_depth":        5,
    "max_leaves":       30,
    "min_child_weight": 13.82,
    "colsample_bytree": 0.5128,
    "colsample_bylevel":0.4462,
    "subsample":        0.9410,
    "gamma":            4.62e-05,
    "reg_alpha":        0.001336,
    "reg_lambda":       0.007528,
    # best_iteration: 977
    # CV AUC: 0.916671
}
```

**LightGBM best params** (from stacking notebook):
```python
{
    "objective":         "binary",
    "metric":            "auc",
    "boosting_type":     "gbdt",
    "learning_rate":     0.018,
    "num_leaves":        84,
    "max_depth":         10,
    "feature_fraction":  0.65,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "lambda_l1":         0.019,
    "lambda_l2":         0.006,
}
# Rounds: 800
```

---

### Technique Tier 3: Feature Engineering

#### Layer 1 — Proven Numeric Row-Stats (8 features, consistently helps)

```python
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df['num_sum']  = df[num_cols].sum(axis=1)
df['num_mean'] = df[num_cols].mean(axis=1)
df['num_std']  = df[num_cols].std(axis=1)
df['num_max']  = df[num_cols].max(axis=1)
df['num_min']  = df[num_cols].min(axis=1)

# Domain interactions
df['Average_Monthly_Actual'] = df['TotalCharges'] / (df['tenure'] + 1e-5)
df['Monthly_diff']           = df['MonthlyCharges'] - df['Average_Monthly_Actual']
df['Monthly_ratio']          = df['MonthlyCharges'] / (df['Average_Monthly_Actual'] + 1e-5)
```

#### Layer 2 — EDA-Driven Domain Features (7 features, from nb17)

```python
# Feature 1: Security services count (non-linear U-shape: 1 service = WORSE than 0)
# Churn rates: 0→29.8%, 1→38.9%, 2→23.8%, 3→12.4%, 4→5.3%
df['n_security_services'] = (
    (df['OnlineSecurity']   == 'Yes').astype(int) +
    (df['OnlineBackup']     == 'Yes').astype(int) +
    (df['DeviceProtection'] == 'Yes').astype(int) +
    (df['TechSupport']      == 'Yes').astype(int)
)

# Feature 2: Fiber optic + zero security (2.40x churn lift!)
df['is_fiber_no_support'] = (
    (df['InternetService']    == 'Fiber optic') &
    (df['n_security_services'] == 0)
).astype(int)

# Feature 3: New customer on month-to-month (1.94x lift)
df['is_new_mtm'] = (
    (df['tenure']   <= 12) &
    (df['Contract'] == 'Month-to-month')
).astype(int)

# Feature 4: Electronic check + month-to-month (2.02x lift)
df['is_echeck_mtm'] = (
    (df['PaymentMethod'] == 'Electronic check') &
    (df['Contract']      == 'Month-to-month')
).astype(int)

# Feature 5: High charges + month-to-month
df['is_high_charge_mtm'] = (
    (df['MonthlyCharges'] > df['MonthlyCharges'].median()) &
    (df['Contract']       == 'Month-to-month')
).astype(int)

# Feature 6: Senior citizen alone (no partner, no dependents)
df['is_senior_alone'] = (
    (df['SeniorCitizen'] == 1) &
    (df['Partner']       == 'No') &
    (df['Dependents']    == 'No')
).astype(int)

# Feature 7: Contract ordinal encoding
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['contract_ord'] = df['Contract'].map(contract_map)
```

#### Layer 3 — Rich Feature Engineering (23 new features, from nb12)

```python
# Service binary flags + counts
service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for c in service_cols:
    df[f"{c}_bin"] = (df[c] == 'Yes').astype(int)

df['n_services_total'] = df[[f"{c}_bin" for c in service_cols]].sum(axis=1)
df['n_streaming']      = df['StreamingTV_bin'] + df['StreamingMovies_bin']
df['n_security']       = (df['OnlineSecurity_bin'] + df['OnlineBackup_bin']
                          + df['DeviceProtection_bin'] + df['TechSupport_bin'])
df['has_internet']     = (df['InternetService'] != 'No').astype(int)
df['has_phone']        = (df['PhoneService'] == 'Yes').astype(int)

# Payment / billing
df['is_autopay']            = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
df['is_paperless_autopay']  = ((df['PaperlessBilling'] == 'Yes') & (df['is_autopay'] == 1)).astype(int)

# Expected vs actual total charges (detects discounts/arrears)
df['expected_total']    = df['MonthlyCharges'] * df['tenure']
df['charge_gap']        = df['TotalCharges'] - df['expected_total']
df['charge_consistency']= df['TotalCharges'] / (df['expected_total'] + 1e-5)

# Tenure segments
df['is_new_customer']    = (df['tenure'] <= 6).astype(int)
df['is_mature_customer'] = (df['tenure'] > 48).astype(int)
df['tenure_bin']         = pd.cut(df['tenure'], bins=[0,6,12,24,48,72], labels=[0,1,2,3,4]).astype(int)

# Log transforms
df['log_tenure']         = np.log1p(df['tenure'])
df['log_TotalCharges']   = np.log1p(df['TotalCharges'])
df['log_MonthlyCharges'] = np.log1p(df['MonthlyCharges'])

# Interaction
df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']
df['senior_alone']     = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
```

**Low-signal features to drop** (< 2pp churn difference): `gender`, `PhoneService`

---

### Technique Tier 4: Scale Pos Weight + Lossguide Growth Policy

Used in nb10+ as a new architectural direction:

```python
# Auto-compute class imbalance weight
churn_rate    = y.mean()                            # 0.2252
scale_pos_w   = (1 - churn_rate) / churn_rate       # ≈ 3.44

base_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "tree_method":      "hist",
    "grow_policy":      "lossguide",   # leaf-wise, like LightGBM
    "scale_pos_weight": scale_pos_w,   # upweight minority class
    "max_leaves":       50,            # controls leaf-wise tree
    "max_depth":        6,
    ...
}
```

**Why lossguide?** Produces asymmetric trees focused on hardest splits. Combines XGBoost's AUC-oriented training with LightGBM-style leaf-wise growth. New Optuna search params: `grow_policy`, `max_leaves`, `max_bin`.

---

### Technique Tier 5: Full Data Retrain Trick

K-Fold trains on `(K-1)/K` of data. To use all data for inference:

```python
# Scale best_round from K-fold to full dataset
full_rounds = int(best_round * N_SPLITS / (N_SPLITS - 1))
# With 10 folds: full_rounds = best_round * 10/9 ≈ 1.11× best_round

full_model = xgb.train(
    {**best_params, 'seed': 42},
    dtrain_full,         # all training data
    num_boost_round=full_rounds
)
full_preds = full_model.predict(dtest)
```

Blend the ensemble preds with full-data retrain preds (50/50) for the final submission.

---

### Technique Tier 6: Original Data Augmentation

The IBM Telco dataset (7,043 rows, same schema) is available as `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Adding it as weighted auxiliary training data:

```python
# Original data has same feature schema
# Churn rate differs: playground=22.5%, original=26.5%

ORIG_WEIGHT = 0.3   # tunable (tried 0.1–0.5)

# Inside each fold, append original data with sample_weight
X_fold_aug = pd.concat([X.iloc[tr_idx], X_orig], axis=0)
y_fold_aug = np.concatenate([y.iloc[tr_idx].values, y_orig.values])
w_fold_aug = np.concatenate([
    np.ones(len(tr_idx)),
    np.full(len(X_orig), ORIG_WEIGHT)
])

dtrain = xgb.DMatrix(X_fold_aug, label=y_fold_aug, weight=w_fold_aug)
```

**Impact**: Provides slight regularization. Helps most when the synthetic data diverges from the true distribution. Currently incorporated in nb17 (EDA FE pipeline).

---

### Technique Tier 7: Two-Level Stacking (nb14)

**Architecture**: 3 base models → 2 meta-models

**Layer 1 — Base Models**:
| ID | Model | Key Features | CV Folds |
|----|-------|-------------|----------|
| BM1 | XGBoost lossguide + scale_pos_weight | Label-encoded features | 10-fold |
| BM2 | LightGBM GBDT tuned | Label-encoded features | 10-fold |
| BM3 | XGBoost + OOF Target Encoding | Target-encoded categoricals | 10-fold |

**Layer 2 — Meta-Models**:
```python
# Meta-features: OOF preds from BM1, BM2, BM3
META_TRAIN = np.column_stack([oof1, oof2, oof3])

# Meta-model A: Logistic Regression (simple, interpretable)
lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(StandardScaler().fit_transform(META_TRAIN), y)

# Meta-model B: XGBoost depth-2 (shallow, prevents overfit)
xgb_meta_params = {"max_depth": 2, "learning_rate": 0.05, ...}

# Final: weighted average of meta-model outputs
```

**OOF Correlation** (key diversity metric — lower = better stacking):
```
                XGB-scaledpos  LGB-tuned  XGB-targetenc
XGB-scaledpos   1.00000        0.99998    0.99997
LGB-tuned       0.99998        1.00000    0.99998
XGB-targetenc   0.99997        0.99998    1.00000
```
Very high correlation (>0.999) means these models are near-identical in ranking behavior — stacking adds minimal benefit over simple blending for this dataset.

---

### Technique Tier 8: OOF Target Encoding (nb13)

Apply smoothed Micci-Barreca (2001) target encoding inside each CV fold to prevent leakage:

```python
class TargetEncoder:
    """Smoothed target encoder."""
    def __init__(self, cols, alpha=10):
        # alpha=10: moderate smoothing; rare categories pulled toward global mean
        self.cols, self.alpha = cols, alpha

    def fit(self, X, y):
        self.global_mean = float(y.mean())
        for c in self.cols:
            agg = pd.DataFrame({'y': y}).assign(cat=X[c]).groupby('cat')['y'].agg(['sum','count'])
            smoothed = (agg['sum'] + self.alpha * self.global_mean) / (agg['count'] + self.alpha)
            self.maps[c] = smoothed.to_dict()

    # Usage: fit only on training fold, transform both train and val
```

Apply to all 15 categorical columns with `alpha=10`. OOF safety: re-fit encoder inside each fold — do not fit globally (introduces leakage).

---

### Technique Tier 9: Pseudo-Labeling (nb15)

Use high-confidence test predictions as additional training labels:

```python
# Config 1: conservative
{"thresh_hi": 0.90, "thresh_lo": 0.10, "pseudo_weight": 0.5}

# Config 2: moderate
{"thresh_hi": 0.95, "thresh_lo": 0.05, "pseudo_weight": 0.5}

# Pseudo dataset construction
mask_churn    = test_preds > thresh_hi   # high-confidence churners
mask_no_churn = test_preds < thresh_lo   # high-confidence retainers

pseudo_X = test[mask_churn | mask_no_churn]
pseudo_y = (test_preds[mask_churn | mask_no_churn] > thresh_hi).astype(int)
pseudo_w = np.full(len(pseudo_X), pseudo_weight)

# Train on: real_train_fold + pseudo_test
# Validate on: real_val_fold only (no pseudo in validation → no leakage)
```

---

### Technique Tier 10: Multi-Model Blending Strategy

**What works**:
- XGB × 2 + LGB × 1.5 + XGB-100n × 1 (custom weighted) → **0.91433**
- Rank averaging tied with probability averaging when scales are aligned
- Simple blending often matches or beats stacking for this dataset

**What doesn't work**:
- Score-weighted blending when a weak model (CatBoost) is included
- Equal-weight blending when one model is significantly weaker

**Rank averaging code**:
```python
from scipy.stats import rankdata

def rank_average(preds_list):
    ranks = [rankdata(p) / len(p) for p in preds_list]
    return np.mean(ranks, axis=0)
```

**Blend notebook pattern** (nb08):
```python
CUSTOM_WEIGHTS = {
    'XGB-50t-FE':     2.0,
    'XGB-100t-6fold': 2.0,
    'LGB-100n':       1.5,
    'XGB-100n':       1.0,
    'CatBoost':       0.0,   # excluded — lower score
}
```

---

## 4. Code Templates

### Template A: Standard Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TARGET = 'Churn'

def load_and_preprocess(data_dir: str):
    """Standard preprocessing for PS S6E3."""
    train = pd.read_csv(f'{data_dir}/train.csv')
    test  = pd.read_csv(f'{data_dir}/test.csv')

    # Encode target
    train[TARGET] = train[TARGET].map({'Yes': 1, 'No': 0})

    # Compute class imbalance weight
    churn_rate    = train[TARGET].mean()
    scale_pos_w   = round((1 - churn_rate) / churn_rate, 4)  # ≈ 3.44

    # Joint preprocessing to avoid category mismatch
    combined  = pd.concat([train.drop(TARGET, axis=1), test]).reset_index(drop=True)
    train_idx = range(len(train))
    test_idx  = range(len(train), len(train) + len(test))

    num_cols = combined.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in combined.select_dtypes(include='object').columns if c != 'id']

    for c in num_cols: combined[c].fillna(combined[c].median(), inplace=True)
    for c in cat_cols: combined[c].fillna('Missing', inplace=True)

    le = LabelEncoder()
    for c in cat_cols:
        combined[c] = le.fit_transform(combined[c].astype(str))

    return combined, train_idx, test_idx, train[TARGET], scale_pos_w
```

### Template B: Feature Engineering (Proven 8 + EDA 7)

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline. Input: raw df (before label encoding)."""
    d = df.copy()
    num_base = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Proven 8 (numeric row-stats + domain interactions)
    d['num_sum']   = d[num_base].sum(axis=1)
    d['num_mean']  = d[num_base].mean(axis=1)
    d['num_std']   = d[num_base].std(axis=1)
    d['num_max']   = d[num_base].max(axis=1)
    d['num_min']   = d[num_base].min(axis=1)
    d['Average_Monthly_Actual'] = d['TotalCharges'] / (d['tenure'] + 1e-5)
    d['Monthly_diff']           = d['MonthlyCharges'] - d['Average_Monthly_Actual']
    d['Monthly_ratio']          = d['MonthlyCharges'] / (d['Average_Monthly_Actual'] + 1e-5)

    # EDA-driven 7 (domain knowledge features)
    d['n_security_services'] = (
        (d['OnlineSecurity']   == 'Yes').astype(int) +
        (d['OnlineBackup']     == 'Yes').astype(int) +
        (d['DeviceProtection'] == 'Yes').astype(int) +
        (d['TechSupport']      == 'Yes').astype(int)
    )
    d['is_fiber_no_support'] = ((d['InternetService'] == 'Fiber optic') &
                                 (d['n_security_services'] == 0)).astype(int)
    d['is_new_mtm']          = ((d['tenure'] <= 12) &
                                 (d['Contract'] == 'Month-to-month')).astype(int)
    d['is_echeck_mtm']       = ((d['PaymentMethod'] == 'Electronic check') &
                                 (d['Contract'] == 'Month-to-month')).astype(int)
    d['is_high_charge_mtm']  = ((d['MonthlyCharges'] > d['MonthlyCharges'].median()) &
                                 (d['Contract'] == 'Month-to-month')).astype(int)
    d['is_senior_alone']     = ((d['SeniorCitizen'] == 1) &
                                 (d['Partner'] == 'No') &
                                 (d['Dependents'] == 'No')).astype(int)
    d['contract_ord']        = d['Contract'].map(
        {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    )

    return d
```

### Template C: Multi-Seed Ensemble Core Loop

```python
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

def run_multiseed_ensemble(
    X, y, X_test, params, n_rounds,
    seeds=(42, 2024, 777, 1337, 999),
    n_splits=10
):
    """
    Multi-seed K-fold ensemble. Returns (oof_preds, test_preds, fold_auc_log).
    """
    total_models = len(seeds) * n_splits
    test_preds_sum = np.zeros(len(X_test))
    oof_sum        = np.zeros(len(X))
    auc_log        = []

    dtest = xgb.DMatrix(X_test)

    for seed in tqdm(seeds, desc='Seeds'):
        skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        seed_oof = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_tr,  label=y_tr)
            dval   = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                {**params, 'seed': seed + fold},
                dtrain,
                num_boost_round = n_rounds,
                evals           = [(dval, 'val')],
                early_stopping_rounds = 100,
                verbose_eval    = False,
            )

            val_preds = model.predict(dval)
            fold_auc  = roc_auc_score(y_val, val_preds)
            auc_log.append(fold_auc)

            seed_oof[val_idx]  = val_preds
            test_preds_sum    += model.predict(dtest)

        oof_sum += seed_oof

    oof_preds   = oof_sum        / len(seeds)
    test_preds  = test_preds_sum / total_models
    final_auc   = roc_auc_score(y, oof_preds)

    print(f'Global OOF AUC: {final_auc:.5f}')
    print(f'Fold AUC mean: {np.mean(auc_log):.5f} ± {np.std(auc_log):.5f}')

    return oof_preds, test_preds, auc_log
```

### Template D: Optuna Tuning with xgb.cv()

```python
import optuna
import xgboost as xgb

def tune_xgb(X, y, n_trials=50, n_cv_folds=5, scale_pos_w=3.44, use_gpu=False):
    """Optuna tuning using xgb.cv() per trial — robust and no single-split overfitting."""

    base_params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'tree_method':      'hist',
        'device':           'cuda' if use_gpu else 'cpu',
        'grow_policy':      'lossguide',
        'scale_pos_weight': scale_pos_w,
        'verbosity':        0,
    }
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    def objective(trial):
        params = {
            **base_params,
            'max_depth':         trial.suggest_int('max_depth', 3, 10),
            'max_leaves':        trial.suggest_int('max_leaves', 15, 127),
            'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'min_child_weight':  trial.suggest_float('min_child_weight', 1, 20),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
            'gamma':             trial.suggest_float('gamma', 1e-8, 5.0, log=True),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        cv = xgb.cv(
            params, dtrain,
            num_boost_round=1000,
            nfold=n_cv_folds,
            stratified=True,
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        trial.set_user_attr('best_iteration', int(cv['test-auc-mean'].idxmax()))
        return cv['test-auc-mean'].max()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_params = {**base_params, **best.params}
    best_rounds  = best.user_attrs['best_iteration'] + 1

    print(f'Best CV AUC: {best.value:.5f}')
    print(f'Best params: {best.params}')
    print(f'Best rounds: {best_rounds}')

    return best_params, best_rounds
```

### Template E: Full Data Retrain Trick

```python
def full_data_retrain(X, y, best_params, best_round, n_splits=10):
    """Retrain on 100% training data with scaled round count."""
    full_rounds = int(best_round * n_splits / (n_splits - 1))
    print(f'CV best_round: {best_round}  |  Full data rounds: {full_rounds}')

    dtrain_full = xgb.DMatrix(X, label=y, enable_categorical=True)
    full_model  = xgb.train(
        {**best_params, 'seed': 42},
        dtrain_full,
        num_boost_round=full_rounds,
        verbose_eval=False,
    )
    return full_model
```

### Template F: Rank Averaging Blend

```python
from scipy.stats import rankdata

def rank_average_blend(submissions: list, weights: list = None) -> np.ndarray:
    """
    Blend multiple prediction arrays using rank averaging.
    More robust than probability averaging when scale differs between models.
    """
    n = len(submissions[0])
    if weights is None:
        weights = [1.0] * len(submissions)

    ranks = [rankdata(p) / n for p in submissions]
    blended = np.average(ranks, weights=weights, axis=0)
    return blended
```

---

## 5. Best Practices for This Competition

### Data Preprocessing
1. **Joint label encoding**: Combine train + test before fitting LabelEncoder — prevents unseen category errors on test set
2. **Fill numeric NaN with median, categorical NaN with 'Missing'**
3. **Do not scale features for tree models** (XGBoost, LightGBM, CatBoost are scale-invariant)
4. **Apply feature engineering before encoding** — FE relies on original string values (e.g., `df['InternetService'] == 'Fiber optic'`)

### Model Selection
1. **XGBoost and LightGBM are the workhorses** — CatBoost is slower and ~0.001 lower AUC on this dataset
2. **`grow_policy='lossguide'` + `max_leaves`** — XGBoost with leaf-wise growth matches LightGBM's architecture
3. **`scale_pos_weight ≈ 3.44`** — corrects class imbalance; tune around ±50% of this value

### Cross-Validation
1. **Always use StratifiedKFold** — 22.5% churn rate requires stratification to ensure balanced folds
2. **N_SPLITS=10 is the sweet spot** — 5-fold gives noisier estimates; 20-fold adds compute without benefit
3. **Seed diversity**: 5 seeds × 10 folds = 50 models is the baseline ensemble size
4. **Use same seeds for all models** when building stacks — enables valid OOF comparison

### Feature Engineering Priorities
1. **Contract type is the single strongest predictor** (MTM=42% vs Two year=1% churn)
2. **Internet service tier** is the second strongest (Fiber=41.5% vs DSL=10.3%)
3. **Payment method** shows strong signal (Electronic check=48.9%)
4. **Interaction terms matter**: fiber+no security, echeck+MTM, new customer+MTM capture non-linear signals
5. **Drop `gender` and `PhoneService`** — < 2pp churn difference, add noise

### CatBoost Gotcha (GPU)
```python
# NEVER use colsample_bylevel (maps to 'rsm') with GPU + binary loss
if not USE_GPU:
    params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', ...)
# Causes crash: "rsm not supported for non-pairwise on GPU"
```

### Tuning
1. **Use `xgb.cv()` / `lgb.cv()` inside each Optuna trial**, not a single split
2. **50–100 trials is sufficient** — plateau observed beyond 50–100 trials
3. **Enable `early_stopping_rounds=50`** within CV to find optimal round count
4. **`max_bin` tuning** (128–512) is part of nb10+ search space — affects histogram resolution

### Blending
1. **XGB + LGB blend** consistently outperforms single best model (+0.0002 AUC)
2. **Rank averaging ≈ probability averaging** when model probability scales are well-calibrated
3. **Exclude weak models from blend** — CatBoost inclusion reduced score despite its standalone 0.91340
4. **Weight by public LB score** only if the weak model is confidently lower

### What Does Not Work
- CatBoost: slower, lower AUC (0.91340 vs 0.91417 for XGBoost best)
- Tuning beyond 100 Optuna trials: plateau effect confirmed
- Score-weighted blend when a weak model (CatBoost) is included
- Single 80/20 split for Optuna objective (noisy, overfit to split)

---

## 6. Competition-Specific Patterns

### Why This Dataset Is "Easy"
The synthetic data is generated directly from the IBM Telco dataset (7,043 rows → 594,194 rows). This means:
- Very high signal-to-noise ratio
- Most models converge to similar AUC (0.912–0.915 range)
- The top of the leaderboard is compressed (marginal gains from extra complexity)
- Simple GBM ensembles dominate; neural networks and AutoML provide no clear advantage here

### Leaderboard Score Distribution (Estimated)
Based on the competition's synthetic data nature and score patterns:
- Top 1–5: likely ~0.916–0.918 (aggressive ensembling + novel FE)
- Top 5–50: likely ~0.914–0.916 (standard GBM ensemble + good FE)
- Top 50–200: likely ~0.912–0.914 (standard GBM, basic FE)
- Median: ~0.910 (single tuned model)

### General 2025 Kaggle Playground Winning Patterns
1. **72+ model ensembles** (Chris Deotte's approach: 3-level stacking)
2. **RAPIDS cuML/cuDF** for GPU-accelerated feature engineering
3. **AutoGluon** dominates entry-level; custom stacks dominate top-10
4. **Consistent K-folds across all base models** (prerequisite for valid OOF stacking)
5. **TabPFN** (prior-data fitted networks) useful for small-to-medium tabular datasets
6. **Groupby aggregations** as FE: `groupby(COL1)[COL2].agg(STAT)` — particularly powerful for datasets with natural groupings

---

## 7. Next Experiments (Prioritized)

| Priority | Experiment | Expected Gain | Status |
|---------|-----------|--------------|--------|
| 1 | Submit nb17 (EDA FE + orig data) | +0.0002–0.0005 | Pending submission |
| 2 | Submit nb15 pseudo-labeling (pl_90_w05, pl_95_w05) | +0.0001–0.0003 | Pending |
| 3 | Blend nb17 + nb09 (LGB+FE) + nb10 (XGB scaledpos) | +0.0002–0.0004 | Pending |
| 4 | Submit nb14 stacking variants (stack_lr, stack_xgb) | +0.0001–0.0002 | Pending |
| 5 | Test CatBoost native categorical mode (no GPU colsample_bylevel) | baseline | Not tried |
| 6 | Neural network (TabNet / MLP with embedding) | uncertain | Not tried |
| 7 | AutoGluon with 9h runtime | uncertain | Not tried |
| 8 | Target encoding alpha sweep (5, 10, 25, 50) | +0.0001 | Partial |

---

## 8. Metadata

| Field | Value |
|-------|-------|
| Knowledge base version | v1.0 |
| Compiled from | 18 local notebooks (nb00–nb17) + web research |
| Local best score | **0.91433** (blend_custom_rank) |
| Best CV AUC achieved | **0.916671** (EDA FE + XGB, nb17 tuning trial #19) |
| Competition platform | Kaggle Playground Series Season 6 Episode 3 |
| Competition ID | `playground-series-s6e3` |
| Data generated from | IBM Telco Customer Churn dataset (WA_Fn-UseC_) |
| Notebooks analyzed | nb00 (EDA), nb06 (XGB multiseed), nb07 (CatBoost), nb08 (blend), nb09 (LGB), nb10 (scaledpos), nb12 (rich FE), nb13 (target enc), nb14 (stacking), nb15 (pseudo-label), nb17 (EDA FE) |
| Date | 2026-03-27 |

---

*Sources consulted*:
- Local notebook files: `/Users/pravintakpire/datascience/KAGGLE_COMPETITION/Predict_Customer_Churn/notebooks/`
- Tuning log: `xgb_eda_fe_tuning_results.csv` (50 trials, best CV AUC=0.916671)
- Run log: `logs/run_nb17.log`
- Competition CLAUDE.md with submission score history
- Web: [Kaggle PS S6E3](https://www.kaggle.com/competitions/playground-series-s6e3), [Kaggle Playground 2025 Patterns](https://medium.com/@gauurab/kaggle-playground-how-top-competitors-actually-win-in-2025-c75d4b380bb5), [S6E2 1st Place](https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t)
