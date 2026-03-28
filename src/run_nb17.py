import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna, warnings, subprocess, os, gc
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Environment ───────────────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")
def has_gpu():
    try: return subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0
    except: return False

USE_GPU  = IS_KAGGLE or has_gpu()
DEVICE   = "cuda" if USE_GPU else "cpu"
DATA_DIR = "/kaggle/input/playground-series-s6e3/" if IS_KAGGLE else "data/"
ORIG_DIR = "/kaggle/input/telco-customer-churn/"   if IS_KAGGLE else "data/"
SUB_DIR  = "/kaggle/working/"                       if IS_KAGGLE else "./"

print(f"Environment : {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"GPU         : {'Enabled ✓' if USE_GPU else 'CPU only'}")
print(f"XGBoost     : {xgb.__version__}")

# ── Settings ──────────────────────────────────────────────────────────────────
RUN_TUNING   = True
N_TRIALS     = 50
N_CV_FOLDS   = 5
SEEDS        = [42, 2024, 777, 1337, 999]
N_SPLITS     = 10
TOTAL_MODELS = len(SEEDS) * N_SPLITS
TARGET       = "Churn"
ORIG_WEIGHT  = 0.0    # SET TO 0 — isolate EDA features effect without original data noise

PRECOMPUTED_PARAMS = {
    "learning_rate":     0.02,
    "max_depth":         6,
    "max_leaves":        50,
    "min_child_weight":  5,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "colsample_bylevel": 0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "gamma":             0.05,
}


# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
orig  = pd.read_csv(os.path.join(ORIG_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

print(f"Playground train : {train.shape}")
print(f"Playground test  : {test.shape}")
print(f"Original (IBM)   : {orig.shape}")

# ── Fix original dataset quirks ───────────────────────────────────────────────
# TotalCharges is stored as string in IBM dataset (spaces = missing)
orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")

# Drop customerID — not in playground data
if "customerID" in orig.columns:
    orig = orig.drop(columns=["customerID"])

# Encode targets
train[TARGET] = (train[TARGET] == "Yes").astype(int)
orig[TARGET]  = (orig[TARGET]  == "Yes").astype(int)

churn_rate  = train[TARGET].mean()
scale_pos_w = round((1 - churn_rate) / churn_rate, 4)
print(f"\nPlayground churn rate : {churn_rate:.4f}")
print(f"Original   churn rate : {orig[TARGET].mean():.4f}")
print(f"scale_pos_weight      : {scale_pos_w}")
print(f"Original TotalCharges NaNs fixed: {orig['TotalCharges'].isna().sum()} rows → median-filled later")

# Submission template
test_ids = test[["id"]].copy()

def build_features(df):
    """
    EDA-driven feature engineering.
    Builds on proven 8 features from nb10 + 6 domain interaction features from EDA.
    Input df: raw (pre-label-encoding), all categoricals as strings.
    """
    d = df.copy()

    # ── PROVEN FE (nb10) ──────────────────────────────────────────────────────
    num_base = ["tenure", "MonthlyCharges", "TotalCharges"]
    d["num_sum"]  = d[num_base].sum(axis=1)
    d["num_mean"] = d[num_base].mean(axis=1)
    d["num_std"]  = d[num_base].std(axis=1)
    d["num_max"]  = d[num_base].max(axis=1)
    d["num_min"]  = d[num_base].min(axis=1)
    d["Average_Monthly_Actual"] = d["TotalCharges"] / (d["tenure"] + 1e-5)
    d["Monthly_diff"]           = d["MonthlyCharges"] - d["Average_Monthly_Actual"]
    d["Monthly_ratio"]          = d["MonthlyCharges"] / (d["Average_Monthly_Actual"] + 1e-5)

    # ── EDA FEATURE 1: Security services count ────────────────────────────────
    # Non-linear: 1 service is WORSE than 0 (toe-dipper effect)
    # 0→29.8%, 1→38.9%, 2→23.8%, 3→12.4%, 4→5.3% churn
    d["n_security_services"] = (
        (d["OnlineSecurity"]  == "Yes").astype(int) +
        (d["OnlineBackup"]    == "Yes").astype(int) +
        (d["DeviceProtection"]== "Yes").astype(int) +
        (d["TechSupport"]     == "Yes").astype(int)
    )

    # ── EDA FEATURE 2: Fiber optic + zero security (2.40x lift) ──────────────
    d["is_fiber_no_support"] = (
        (d["InternetService"]  == "Fiber optic") &
        (d["n_security_services"] == 0)
    ).astype(int)

    # ── EDA FEATURE 3: New customer on month-to-month (1.94x lift) ───────────
    d["is_new_mtm"] = (
        (d["tenure"]   <= 12) &
        (d["Contract"] == "Month-to-month")
    ).astype(int)

    # ── EDA FEATURE 4: Electronic check + month-to-month (2.02x lift) ────────
    d["is_echeck_mtm"] = (
        (d["PaymentMethod"] == "Electronic check") &
        (d["Contract"]      == "Month-to-month")
    ).astype(int)

    # ── EDA FEATURE 5: High charges + no lock-in (1.99x lift) ────────────────
    d["is_high_charge_mtm"] = (
        (d["MonthlyCharges"] > 70) &
        (d["Contract"]       == "Month-to-month")
    ).astype(int)

    # ── EDA FEATURE 6: Senior living alone (1.85x lift) ──────────────────────
    d["is_senior_alone"] = (
        (d["SeniorCitizen"] == 1) &
        (d["Partner"]       == "No") &
        (d["Dependents"]    == "No")
    ).astype(int)

    # ── EDA INSIGHT: Contract ordinal (captures ordering) ────────────────────
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    d["contract_ord"] = d["Contract"].map(contract_map).fillna(0).astype(int)

    return d

print("Feature engineering function defined.")
print("New EDA features: n_security_services, is_fiber_no_support, is_new_mtm,")
print("                  is_echeck_mtm, is_high_charge_mtm, is_senior_alone, contract_ord")


# ── Columns to drop (near-zero signal from EDA) ──────────────────────────────
DROP_COLS = ["gender", "PhoneService"]   # spread < 2pp in EDA
print(f"Dropping low-signal cols: {DROP_COLS}")

# Separate targets before combining
y_train = train[TARGET].copy()
y_orig  = orig[TARGET].copy()

# Joint preprocessing: playground train + test (avoid category mismatch)
playground_features = train.drop(columns=[TARGET])
combined = pd.concat([playground_features, test], axis=0).reset_index(drop=True)
train_idx = range(len(train))
test_idx  = range(len(train), len(train) + len(test))

# Also preprocess original separately (same pipeline)
orig_features = orig.drop(columns=[TARGET])

# Fill nulls on combined
num_cols = combined.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in combined.select_dtypes(include="object").columns
            if c not in ["id"] + DROP_COLS]

for c in num_cols:
    combined[c].fillna(combined[c].median(), inplace=True)
for c in cat_cols:
    combined[c].fillna("Missing", inplace=True)

# Fill nulls on original
for c in num_cols:
    if c in orig_features.columns:
        orig_features[c].fillna(orig_features[c].median(), inplace=True)
for c in cat_cols:
    if c in orig_features.columns:
        orig_features[c].fillna("Missing", inplace=True)

print(f"Categorical cols after drop: {cat_cols}")
print(f"Numeric cols: {num_cols}")


# Apply FE to combined playground data
combined_fe = build_features(combined)
orig_fe     = build_features(orig_features)

# Drop low-signal cols
combined_fe = combined_fe.drop(columns=DROP_COLS, errors="ignore")
orig_fe     = orig_fe.drop(columns=DROP_COLS, errors="ignore")

# Label encode — fit on combined (playground), apply to both
FEATURES_ALL = [c for c in combined_fe.columns if c not in ["id", TARGET]]

le = LabelEncoder()
for c in cat_cols:
    if c in combined_fe.columns:
        # Fit on combined values + original values to avoid unseen categories
        all_vals = pd.concat([combined_fe[c], orig_fe[c]]).astype(str)
        le.fit(all_vals)
        combined_fe[c] = le.transform(combined_fe[c].astype(str))
        orig_fe[c]     = le.transform(orig_fe[c].astype(str))

FEATURES = [c for c in combined_fe.columns if c not in ["id", TARGET]]

# Split back
train_fe = combined_fe.iloc[train_idx].copy()
test_fe  = combined_fe.iloc[test_idx].copy()

X_playground = train_fe[FEATURES]
X_test       = test_fe[FEATURES]
X_orig       = orig_fe[FEATURES]

print(f"Total features   : {len(FEATURES)}")
print(f"X_playground     : {X_playground.shape}")
print(f"X_orig           : {X_orig.shape}")
print(f"X_test           : {X_test.shape}")
print(f"\nFeature list:")
for i, f in enumerate(FEATURES):
    tag = " ◀ NEW (EDA)" if f in ["n_security_services","is_fiber_no_support",
                                    "is_new_mtm","is_echeck_mtm",
                                    "is_high_charge_mtm","is_senior_alone",
                                    "contract_ord"] else           " ◀ PROVEN FE" if f in ["num_sum","num_mean","num_std","num_max","num_min",
                                    "Average_Monthly_Actual","Monthly_diff","Monthly_ratio"] else ""
    print(f"  {i+1:3d}. {f}{tag}")


base_params = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "tree_method":      "hist",
    "device":           DEVICE,
    "grow_policy":      "lossguide",
    "scale_pos_weight": scale_pos_w,
    "verbosity":        0,
    "nthread":          -1,
    "seed":             42,
}
print(f"XGB device={DEVICE}  grow_policy=lossguide  scale_pos_weight={scale_pos_w}")

# Tuning uses playground-only data (clean signal)
dtrain_full = xgb.DMatrix(X_playground, label=y_train)

if RUN_TUNING:
    print(f"\nOptuna: {N_TRIALS} trials × {N_CV_FOLDS}-fold CV")
    pbar = tqdm(total=N_TRIALS, desc="Optuna Trials", unit="trial")
    best_so_far = [0.0]

    def objective(trial):
        params = {
            **base_params,
            "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.1,  log=True),
            "max_depth":         trial.suggest_int(  "max_depth",         3,     9),
            "max_leaves":        trial.suggest_int(  "max_leaves",        15,    127),
            "min_child_weight":  trial.suggest_float("min_child_weight",  1,     20),
            "subsample":         trial.suggest_float("subsample",         0.5,   1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.4,   1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4,   1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-8,  10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-8,  10.0, log=True),
            "gamma":             trial.suggest_float("gamma",             1e-8,  5.0,  log=True),
        }
        cv = xgb.cv(
            params, dtrain_full,
            num_boost_round=1000, nfold=N_CV_FOLDS, stratified=True,
            early_stopping_rounds=50, seed=42, verbose_eval=False,
        )
        score     = cv["test-auc-mean"].max()
        best_iter = int(cv["test-auc-mean"].idxmax())
        trial.set_user_attr("best_iteration", best_iter)
        if score > best_so_far[0]: best_so_far[0] = score
        pbar.set_postfix({"AUC": f"{score:.5f}", "Best": f"{best_so_far[0]:.5f}"})
        pbar.update(1)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    pbar.close()

    best_trial  = study.best_trial
    best_iter   = best_trial.user_attrs.get("best_iteration", 700)
    best_params = {**base_params, **best_trial.params}

    print(f"\n{'='*55}")
    print(f"Best CV AUC    : {best_trial.value:.5f}")
    print(f"Best iteration : {best_iter}")
    for k, v in best_trial.params.items():
        print(f"  {k:25s}: {v}")
    print(f"{'='*55}")

    study.trials_dataframe().to_csv(
        os.path.join(SUB_DIR, "xgb_eda_fe_tuning_results.csv"), index=False)
else:
    print("Using PRECOMPUTED_PARAMS.")
    best_params = {**base_params, **PRECOMPUTED_PARAMS}
    best_iter   = 700


print(f"Multi-Seed Ensemble: {len(SEEDS)} seeds × {N_SPLITS} folds = {TOTAL_MODELS} models")
print(f"Original data weight: {ORIG_WEIGHT}  ({len(X_orig):,} extra rows per fold)")

test_preds_sum = np.zeros(len(X_test))
global_oof_sum = np.zeros(len(X_playground))
fold_auc_log   = []

dtest = xgb.DMatrix(X_test)

# Pre-build original DMatrix features (constant across all folds)
w_orig   = np.full(len(X_orig), ORIG_WEIGHT)
y_orig_v = y_orig.values

outer_bar = tqdm(SEEDS, desc="Seeds", position=0)

for seed in outer_bar:
    skf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(X_playground))

    inner_bar = tqdm(
        enumerate(skf.split(X_playground, y_train)),
        total=N_SPLITS, desc="  Folds", position=1, leave=False,
    )

    for fold, (tr_idx, val_idx) in inner_bar:
        X_tr  = X_playground.iloc[tr_idx]
        y_tr  = y_train.iloc[tr_idx]
        X_val = X_playground.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Combine playground train fold + full original data
        X_tr_comb = pd.concat([X_tr, X_orig],  axis=0).reset_index(drop=True)
        y_tr_comb = np.concatenate([y_tr.values, y_orig_v])
        w_tr_comb = np.concatenate([np.ones(len(X_tr)), w_orig])

        dtrain = xgb.DMatrix(X_tr_comb, label=y_tr_comb, weight=w_tr_comb)
        dval   = xgb.DMatrix(X_val,     label=y_val)

        params = {**best_params, "seed": seed}
        model  = xgb.train(
            params, dtrain,
            num_boost_round       = best_iter + 200,
            evals                 = [(dval, "val")],
            early_stopping_rounds = 100,
            verbose_eval          = False,
        )

        val_preds  = model.predict(dval,  iteration_range=(0, model.best_iteration))
        test_preds = model.predict(dtest, iteration_range=(0, model.best_iteration))

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_auc_log.append(fold_auc)
        seed_oof[val_idx]        = val_preds
        global_oof_sum[val_idx] += val_preds
        test_preds_sum          += test_preds

        inner_bar.set_postfix({"fold_auc": f"{fold_auc:.5f}", "best_iter": model.best_iteration})
        del model, dtrain, dval; gc.collect()

    seed_auc = roc_auc_score(y_train, seed_oof)
    outer_bar.set_postfix({"seed_oof": f"{seed_auc:.5f}"})

global_oof = global_oof_sum / len(SEEDS)
final_auc  = roc_auc_score(y_train, global_oof)
print(f"\n{'='*55}")
print(f"Fold AUC : {np.mean(fold_auc_log):.5f} ± {np.std(fold_auc_log):.5f}")
print(f"OOF AUC  : {final_auc:.5f}")
print(f"{'='*55}")


final_preds = test_preds_sum / TOTAL_MODELS
out         = test_ids.copy()
out[TARGET] = final_preds
out_file    = os.path.join(SUB_DIR, "submission_xgb_eda_fe.csv")
out.to_csv(out_file, index=False)

print(f"Saved → {out_file}")
print(f"Pred range : [{final_preds.min():.4f}, {final_preds.max():.4f}]")
print(f"Pred mean  : {final_preds.mean():.4f}")
print(out.head())


import subprocess

r = subprocess.run([
    "kaggle", "competitions", "submit",
    "-c", "playground-series-s6e3",
    "-f", os.path.join(SUB_DIR, "submission_xgb_eda_fe.csv"),
    "-m", ("XGB scaledpos+lossguide | EDA FE: fiber_no_support+echeck_mtm+"
           "high_charge_mtm+new_mtm+senior_alone+n_sec | orig_data w=0.3 | 50 models"),
], capture_output=True, text=True)
print(r.stdout)
if r.stderr: print("STDERR:", r.stderr)

