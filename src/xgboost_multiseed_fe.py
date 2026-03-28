import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import gc

warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Creates arbitrary aggregations and statistical features on the continuous columns.
    Inspired by strategies from Predict Heart Disease competition.
    """
    df = df.copy()
    
    # Selecting the main numeric columns based on EDA
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # 1. Row Statistics (Aggregates)
    df['num_sum'] = df[num_cols].sum(axis=1)
    df['num_mean'] = df[num_cols].mean(axis=1)
    df['num_std'] = df[num_cols].std(axis=1)
    df['num_max'] = df[num_cols].max(axis=1)
    df['num_min'] = df[num_cols].min(axis=1)
    
    # 2. Domain / Interaction Features
    # Interactions between charges and tenure
    df['Average_Monthly_Actual'] = df['TotalCharges'] / (df['tenure'] + 1e-5) # Add small epsilon to avoid div by zero
    df['Monthly_diff'] = df['MonthlyCharges'] - df['Average_Monthly_Actual']
    df['Monthly_ratio'] = df['MonthlyCharges'] / (df['Average_Monthly_Actual'] + 1e-5)
    
    return df

def main():
    print("Loading Data...")
    DATA_DIR = 'data/'
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    target_col = 'Churn'
    if train_df[target_col].dtype == 'object':
        train_df[target_col] = train_df[target_col].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
        
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    df = pd.concat([train_df, test_df], ignore_index=True)

    features = [c for c in train_df.columns if c not in ['id', target_col, 'is_train']]
    categorical_features = []

    print("Encoding & Preprocessing...")
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
    test_encoded = df[df['is_train'] == 0].reset_index(drop=True)

    X = train_encoded[features]
    y = train_encoded[target_col]
    X_test = test_encoded[features]

    print("Engineering Extended Features...")
    X = engineer_features(X)
    X_test = engineer_features(X_test)
    
    print(f"Final Feature Count: {X.shape[1]}")

    # Explicit category casting for native XGBoost support
    for col in categorical_features:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    # --- ADVANCED HYPERPARAMETERS ---
    # Derived from generalized Optuna runs + Heart disease baselines
    best_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.015,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'tree_method': 'hist',
        # 'device': 'cuda', # Enable if running locally on a GPU
    }

    # --- THE MAGIC SAUCE: MULTI-SEED K-FOLD ENSEMBLE ---
    SEEDS = [42, 2024, 777, 1337, 999] 
    N_SPLITS = 10 # 10 Folds per seed

    num_total_models = len(SEEDS) * N_SPLITS
    print(f"\nStarting Multi-Seed {N_SPLITS}-Fold Ensemble Training.")
    print(f"Using {len(SEEDS)} seeds -> Training {num_total_models} total models...")

    test_preds_sum = np.zeros(len(X_test))
    oof_preds_sum = np.zeros(len(X)) # Initialize overall OOF sum
    
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Processing Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) ---")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        
        seed_oof = np.zeros(len(X)) # OOF for this specific seed
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain_fold = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
            dval_fold = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            
            # Train with early stopping per fold
            model_fold = xgb.train(
                best_params,
                dtrain_fold,
                num_boost_round=3000, 
                evals=[(dtrain_fold, 'train'), (dval_fold, 'validation')],
                early_stopping_rounds=150,
                verbose_eval=False
            )
            
            # Local Validation Predict
            fold_val_preds = model_fold.predict(dval_fold, iteration_range=(0, model_fold.best_iteration + 1))
            seed_oof[val_idx] = fold_val_preds
            oof_preds_sum[val_idx] += fold_val_preds # Add to global OOF
            
            # Accumulate Test Predictions
            fold_test_preds = model_fold.predict(dtest, iteration_range=(0, model_fold.best_iteration + 1))
            test_preds_sum += fold_test_preds
            
            print(f"  Seed {seed} | Fold {fold+1:02d} AUC: {roc_auc_score(y_val, fold_val_preds):.5f}")
            
        seed_auc = roc_auc_score(y, seed_oof)
        print(f"-> Seed {seed} Overall OOF AUC: {seed_auc:.5f}")

    # Evaluate Global Ensemble OOF
    global_oof = oof_preds_sum / len(SEEDS)
    global_auc = roc_auc_score(y, global_oof)
    print(f"\n==========================================")
    print(f"FINAL MULTI-SEED ENSEMBLE OOF AUC: {global_auc:.5f}")
    print(f"==========================================")

    # Average over all trained models for final test submission
    print(f"\nAveraging test predictions across all {num_total_models} models...")
    test_preds = test_preds_sum / num_total_models

    sub[target_col] = test_preds
    out_file = "submission_xgboost_multiseed_fe.csv"
    sub.to_csv(out_file, index=False)
    print(f"\nFinal ensembled submission saved to {out_file}")

if __name__ == "__main__":
    main()
