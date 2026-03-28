import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def main():
    print("Loading data...")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub = pd.read_csv('data/sample_submission.csv')

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
    test_df = df[df['is_train'] == 0].reset_index(drop=True)

    X = train_df[features]
    y = train_df[target]
    X_test = test_df[features]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.018116179124401863,
        'num_leaves': 84,
        'max_depth': 10,
        'feature_fraction': 0.6537059149753749,
        'bagging_fraction': 0.6274157655853843,
        'bagging_freq': 5,
        'lambda_l1': 0.018599555943730934,
        'lambda_l2': 0.006462630559554851,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features, free_raw_data=False)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=0)]
        )
        
        valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
        oof_preds[valid_idx] = valid_preds
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        
        score = roc_auc_score(y_valid, valid_preds)
        scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.5f}")
        
    print(f"\nMean AUC: {np.mean(scores):.5f}")
    print(f"OOF AUC:  {roc_auc_score(y, oof_preds):.5f}")
    
    sub['Churn'] = test_preds
    sub.to_csv('submission_lgbm_tuned.csv', index=False)
    print("Saved tuned submission!")

if __name__ == "__main__":
    main()
