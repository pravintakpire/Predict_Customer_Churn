import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import optuna
import os
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50        # increase for better results (100+ for competition)
N_CV_FOLDS = 5       # folds used inside each Optuna trial
EARLY_STOPPING = 50  # rounds patience inside xgb.cv
MAX_BOOST_ROUNDS = 1000


def engineer_features(df):
    df = df.copy()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df['num_sum']  = df[num_cols].sum(axis=1)
    df['num_mean'] = df[num_cols].mean(axis=1)
    df['num_std']  = df[num_cols].std(axis=1)
    df['num_max']  = df[num_cols].max(axis=1)
    df['num_min']  = df[num_cols].min(axis=1)
    df['Average_Monthly_Actual'] = df['TotalCharges'] / (df['tenure'] + 1e-5)
    df['Monthly_diff']  = df['MonthlyCharges'] - df['Average_Monthly_Actual']
    df['Monthly_ratio'] = df['MonthlyCharges'] / (df['Average_Monthly_Actual'] + 1e-5)
    return df


def load_and_preprocess():
    DATA_DIR = 'data/'
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    target_col = 'Churn'
    if train_df[target_col].dtype == 'object':
        train_df[target_col] = train_df[target_col].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})

    train_df['is_train'] = 1
    test_df['is_train']  = 0
    df = pd.concat([train_df, test_df], ignore_index=True)

    features = [c for c in train_df.columns if c not in ['id', target_col, 'is_train']]
    categorical_features = []

    for col in features:
        if df[col].dtype == 'object':
            categorical_features.append(col)
            le = LabelEncoder()
            df[col] = df[col].fillna('Missing')
            df[col] = le.fit_transform(df[col].astype(str))

    num_features = [c for c in features if c not in categorical_features]
    for col in num_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    train_encoded = df[df['is_train'] == 1].reset_index(drop=True)

    X = engineer_features(train_encoded[features])
    y = train_encoded[target_col]

    for col in categorical_features:
        X[col] = X[col].astype('category')

    return X, y, categorical_features


def objective(trial, dtrain):
    params = {
        'objective':        'binary:logistic',
        'eval_metric':      'auc',
        'tree_method':      'hist',
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'gamma':            trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbosity':        0,
        'seed':             42,
    }

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        nfold=N_CV_FOLDS,
        stratified=True,
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
        seed=42,
    )

    best_auc = cv_result['test-auc-mean'].max()
    best_round = int(cv_result['test-auc-mean'].idxmax())
    trial.set_user_attr('best_round', best_round)
    return best_auc


def main():
    print("Loading and preprocessing data...")
    X, y, categorical_features = load_and_preprocess()
    print(f"Feature count: {X.shape[1]}")

    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    print(f"\nStarting Optuna XGBoost tuning ({N_TRIALS} trials, {N_CV_FOLDS}-fold CV per trial)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dtrain), n_trials=N_TRIALS, show_progress_bar=True)

    best_trial = study.best_trial
    print(f"\n{'='*50}")
    print(f"Best CV AUC : {best_trial.value:.5f}")
    print(f"Best round  : {best_trial.user_attrs.get('best_round', 'N/A')}")
    print(f"\nBest params (copy into xgboost_multiseed_fe.py):")
    print("-" * 50)
    best_params = {
        'objective':         'binary:logistic',
        'eval_metric':       'auc',
        'tree_method':       'hist',
        **best_trial.params
    }
    for k, v in best_params.items():
        print(f"  '{k}': {v},")
    print(f"{'='*50}")

    # Save results to file for reference
    results_df = study.trials_dataframe()
    results_df.to_csv('xgboost_tuning_results.csv', index=False)
    print(f"\nAll trial results saved to xgboost_tuning_results.csv")


if __name__ == "__main__":
    main()
