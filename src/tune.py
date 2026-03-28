import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import optuna
import gc
import sys

def main():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    if train['Churn'].dtype == 'object':
        train['Churn'] = train['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
            
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], ignore_index=True)

    target = 'Churn'
    features = [c for c in train.columns if c not in ['id', target, 'is_train']]
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

    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    X = train_df[features]
    y = train_df[target]

    from sklearn.model_selection import train_test_split
    X_train_full, X_valid_full, y_train_full, y_valid_full = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbose': -1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        train_data = lgb.Dataset(X_train_full, label=y_train_full, categorical_feature=categorical_features, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid_full, label=y_valid_full, reference=train_data, categorical_feature=categorical_features, free_raw_data=False)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        valid_preds = model.predict(X_valid_full, num_iteration=model.best_iteration)
        score = roc_auc_score(y_valid_full, valid_preds)
            
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    print("\nNumber of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
