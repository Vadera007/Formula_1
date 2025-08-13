import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --- Configuration ---
DB_FILE = "f1_predictor.db"
TABLE_NAME = "historical_data"
MODEL_DIR = "Trinity" # Directory to save models

# Model features (ensure this matches your app.py)
numerical_features = [
    'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
    'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
    'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
    'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
    'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
    'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
    'PracticeFastestLap', 'AvgLongRunPace',
    'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance'
]
categorical_features = ['Driver', 'Team', 'StartingTyreCompound', 'TrackType']
FEATURES = numerical_features + categorical_features

def calculate_ndcg(y_true, y_pred, k=None):
    """Calculates Normalized Discounted Cumulative Gain."""
    if k is None: k = len(y_true)
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).sort_values('pred', ascending=False)
    # Negate true relevance because lower RaceFinishPosition is better
    relevance = -df['true'] 
    dcg = np.sum((2**relevance - 1) / np.log2(np.arange(2, k + 2)))
    ideal_relevance = -df['true'].sort_values(ascending=False)
    idcg = np.sum((2**ideal_relevance - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0

def main():
    """Main function to train and save models."""
    print("--- Starting Model Training ---")
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        if df.empty:
            print("Database is empty. Skipping training.")
            return
    except Exception as e:
        print(f"Error loading data from {DB_FILE}: {e}")
        return

    df['race_id'] = df['Year'].astype(str) + '-' + df['RoundNumber'].astype(str)
    num_races = df['race_id'].nunique()
    print(f"Preparing to train on {len(df)} rows covering {num_races} unique races.")

    for col in categorical_features:
        df[col] = df[col].astype('category')

    X = df[FEATURES]
    y = df['RaceFinishPosition']
    groups = df['race_id']
    
    # --- FIX: Calculate group sizes correctly for XGBoost/LightGBM ---
    # This creates an array like [20, 20, 18, 22, ...] where each number
    # is the count of drivers in that race.
    group_sizes = X.groupby(groups).size().to_numpy()

    gkf = GroupKFold(n_splits=3)
    
    # --- Train Final Models on ALL Data for Production ---
    print("\n--- Training Final Models on ALL Data for Production ---")
    
    # --- XGBoost ---
    print("Training XGBoost...")
    # --- FIX: Added enable_categorical=True ---
    prod_xgb = xgb.XGBRanker(objective='rank:ndcg', eval_metric=['ndcg@10'], n_estimators=200, learning_rate=0.1, tree_method='hist', random_state=42, enable_categorical=True)
    prod_xgb.fit(X, y, group=group_sizes, verbose=False)
    joblib.dump(prod_xgb, os.path.join(MODEL_DIR, 'xgb_model.joblib'))
    print("XGBoost model saved.")

    # --- LightGBM ---
    print("Training LightGBM...")
    prod_lgbm = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', n_estimators=200, learning_rate=0.1, random_state=42)
    prod_lgbm.fit(X, y, group=group_sizes)
    joblib.dump(prod_lgbm, os.path.join(MODEL_DIR, 'lgbm_model.joblib'))
    print("LightGBM model saved.")

    # --- CatBoost ---
    print("Training CatBoost...")
    cat_features_indices = [X.columns.get_loc(c) for c in categorical_features]
    prod_cat = cb.CatBoostRanker(iterations=200, learning_rate=0.1, loss_function='YetiRank', eval_metric='NDCG', random_seed=42, verbose=0)
    prod_cat.fit(X, y, group_id=groups, cat_features=cat_features_indices)
    joblib.dump(prod_cat, os.path.join(MODEL_DIR, 'catboost_model.joblib'))
    print("CatBoost model saved.")

    # --- Run Cross-Validation Separately to Get a True Performance Score ---
    print("\n--- Running Cross-Validation for NDCG Scores ---")
    xgb_scores, lgbm_scores, cat_scores, ensemble_scores = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Processing Fold {fold+1}/3...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Get group info for the training and validation sets
        groups_train = X_train.groupby(groups.iloc[train_idx]).size().to_numpy()
        
        # Train temporary models for this fold
        # --- FIX: Added enable_categorical=True ---
        xgb_cv = xgb.XGBRanker(objective='rank:ndcg', random_state=42, enable_categorical=True)
        xgb_cv.fit(X_train, y_train, group=groups_train)
        
        lgbm_cv = lgb.LGBMRanker(objective='lambdarank', random_state=42)
        lgbm_cv.fit(X_train, y_train, group=groups_train)
        
        cat_cv = cb.CatBoostRanker(loss_function='YetiRank', random_seed=42, verbose=0)
        cat_cv.fit(X_train, y_train, group_id=groups.iloc[train_idx], cat_features=cat_features_indices)

        # Predict scores for the validation set
        xgb_pred = xgb_cv.predict(X_val)
        lgbm_pred = lgbm_cv.predict(X_val)
        cat_pred = cat_cv.predict(X_val)
        ensemble_pred = (xgb_pred + lgbm_pred + cat_pred) / 3.0
        
        # Calculate NDCG for this fold
        xgb_scores.append(calculate_ndcg(y_val, xgb_pred))
        lgbm_scores.append(calculate_ndcg(y_val, lgbm_pred))
        cat_scores.append(calculate_ndcg(y_val, cat_pred))
        ensemble_scores.append(calculate_ndcg(y_val, ensemble_pred))

    print("\n--- Average NDCG Scores (from Cross-Validation) ---")
    print(f"XGBoost Average NDCG: {np.mean(xgb_scores):.4f}")
    print(f"LightGBM Average NDCG: {np.mean(lgbm_scores):.4f}")
    print(f"CatBoost Average NDCG: {np.mean(cat_scores):.4f}")
    print(f"**Trinity Ensemble Average NDCG: {np.mean(ensemble_scores):.4f}**")

    print("\n--- Pipeline finished successfully! ---")

if __name__ == "__main__":
    main()






# #TRINITY

# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import ndcg_score
# from xgboost import XGBRanker
# from lightgbm import LGBMRanker
# from catboost import CatBoostRanker
# import joblib
# import sqlite3
# import warnings

# # Suppress the specific UserWarning from scikit-learn about feature names
# warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMRanker was fitted with feature names")


# # --- Configuration ---
# DB_FILE = "f1_predictor.db"
# XGB_MODEL_FILE = "Trinity/xgb_model.joblib"
# LGBM_MODEL_FILE = "Trinity/lgbm_model.joblib"
# CATBOOST_MODEL_FILE = "Trinity/catboost_model.joblib"

# # --- Main Model Training Logic ---
# if __name__ == "__main__":
#     # Create the model directory if it doesn't exist
#     os.makedirs("Trinity", exist_ok=True)

#     if not os.path.exists(DB_FILE):
#         print(f"Error: '{DB_FILE}' not found. Run data_collection.py first.")
#         exit()

#     print(f"Loading data from {DB_FILE}...")
#     conn = sqlite3.connect(DB_FILE)
#     df = pd.read_sql('SELECT * FROM historical_data', conn)
#     conn.close()
    
#     df['EventDate'] = pd.to_datetime(df['EventDate'], utc=True)

#     # --- Feature Lists ---
#     numerical_features = [
#         'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
#         'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
#         'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
#         'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
#         'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
#         'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
#         'PracticeFastestLap', 'AvgLongRunPace',
#         'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance'
#     ]
#     categorical_features = ['Driver', 'Team', 'StartingTyreCompound', 'TrackType']
#     target = 'RaceFinishPosition'

#     df.dropna(subset=[target] + numerical_features, inplace=True)
#     df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
    
#     df[target] = df.groupby(['Year', 'RoundNumber'])[target].transform(lambda x: x.max() - x + 1)
#     df['group_id'] = df.groupby(['Year', 'RoundNumber']).ngroup()

#     X_full = df[numerical_features + categorical_features]
#     y_full = df[target]
#     groups_full = df.groupby(['Year', 'RoundNumber']).size().to_numpy()
#     group_ids_full = df['group_id']
    
#     print(f"Data prepared for ranking with {len(groups_full)} race groups.")

#     # --- Preprocessing ---
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', 'passthrough', numerical_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
#         ], remainder='drop')

#     # --- Model Definitions ---
#     xgb_params = {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'gamma': 0.1, 'min_child_weight': 1}
#     lgbm_params = {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8}
#     catboost_params = {'iterations': 300, 'learning_rate': 0.1, 'depth': 3, 'verbose': 0}

#     xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', XGBRanker(objective='rank:ndcg', **xgb_params, random_state=42))])
#     lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', LGBMRanker(objective='lambdarank', **lgbm_params, random_state=42))])
#     catboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', CatBoostRanker(objective='QueryRMSE', **catboost_params, random_state=42))])

#     # --- Ensemble Evaluation with GroupKFold ---
#     print("\n--- Evaluating Ensemble Performance with GroupKFold Cross-Validation ---")
#     gkf = GroupKFold(n_splits=3)
    
#     xgb_scores_folds, lgbm_scores_folds, catboost_scores_folds, ensemble_scores_folds = [], [], [], []

#     for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y_full, groups=group_ids_full)):
#         print(f"\n--- Processing Fold {fold+1}/3 ---")
#         X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
#         y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
#         train_groups = df.iloc[train_idx].groupby('group_id').size().to_numpy()
        
#         # Train and evaluate each model individually
#         print("Training and Evaluating XGBoost...")
#         xgb_pipeline.fit(X_train, y_train, ranker__group=train_groups)
#         xgb_pred = xgb_pipeline.predict(X_val)
        
#         print("Training and Evaluating LightGBM...")
#         lgbm_pipeline.fit(X_train, y_train, ranker__group=train_groups)
#         lgbm_pred = lgbm_pipeline.predict(X_val)
        
#         print("Training and Evaluating CatBoost...")
#         X_train_cat, X_val_cat = X_train.copy(), X_val.copy()
#         X_train_cat['group_id'] = df.iloc[train_idx]['group_id']
#         catboost_pipeline.fit(X_train_cat, y_train, ranker__group_id=X_train_cat['group_id'])
#         catboost_pred = catboost_pipeline.predict(X_val_cat)
        
#         # Evaluate individual models and the ensemble
#         val_df = pd.DataFrame({'y_val': y_val, 'group_id': group_ids_full.iloc[val_idx],
#                                'xgb_pred': xgb_pred, 'lgbm_pred': lgbm_pred, 'catboost_pred': catboost_pred})
#         val_df['ensemble_pred'] = (val_df['xgb_pred'] + val_df['lgbm_pred'] + val_df['catboost_pred']) / 3.0

#         fold_xgb_scores, fold_lgbm_scores, fold_catboost_scores, fold_ensemble_scores = [], [], [], []
#         for group_id in val_df['group_id'].unique():
#             group_data = val_df[val_df['group_id'] == group_id]
#             if len(group_data) > 1:
#                 fold_xgb_scores.append(ndcg_score([group_data['y_val'].values], [group_data['xgb_pred'].values]))
#                 fold_lgbm_scores.append(ndcg_score([group_data['y_val'].values], [group_data['lgbm_pred'].values]))
#                 fold_catboost_scores.append(ndcg_score([group_data['y_val'].values], [group_data['catboost_pred'].values]))
#                 fold_ensemble_scores.append(ndcg_score([group_data['y_val'].values], [group_data['ensemble_pred'].values]))
        
#         xgb_scores_folds.append(np.mean(fold_xgb_scores))
#         lgbm_scores_folds.append(np.mean(fold_lgbm_scores))
#         catboost_scores_folds.append(np.mean(fold_catboost_scores))
#         ensemble_scores_folds.append(np.mean(fold_ensemble_scores))

#     # --- Train and Save Final Models on ALL Data ---
#     print("\n--- Training Final Models on ALL Data for Production ---")
#     print("Training XGBoost...")
#     xgb_pipeline.fit(X_full, y_full, ranker__group=groups_full)
#     joblib.dump(xgb_pipeline, XGB_MODEL_FILE)
#     print(f"XGBoost model saved to {XGB_MODEL_FILE}")

#     print("Training LightGBM...")
#     lgbm_pipeline.fit(X_full, y_full, ranker__group=groups_full)
#     joblib.dump(lgbm_pipeline, LGBM_MODEL_FILE)
#     print(f"LightGBM model saved to {LGBM_MODEL_FILE}")

#     print("Training CatBoost...")
#     X_full_cat = X_full.copy()
#     X_full_cat['group_id'] = df['group_id']
#     catboost_pipeline.fit(X_full_cat, y_full, ranker__group_id=X_full_cat['group_id'])
#     joblib.dump(catboost_pipeline, CATBOOST_MODEL_FILE)
#     print(f"CatBoost model saved to {CATBOOST_MODEL_FILE}")

#     print("\n--- All models trained and saved successfully! ---")

#     # --- Final Score Summary ---
#     print("\n--- Cross-Validation NDCG Scores ---")
#     print(f"XGBoost Scores per Fold: {[round(s, 4) for s in xgb_scores_folds]}")
#     print(f"LightGBM Scores per Fold: {[round(s, 4) for s in lgbm_scores_folds]}")
#     print(f"CatBoost Scores per Fold: {[round(s, 4) for s in catboost_scores_folds]}")
#     print(f"Trinity Ensemble Scores per Fold: {[round(s, 4) for s in ensemble_scores_folds]}")
#     print("\n--- Average NDCG Scores ---")
#     print(f"XGBoost Average NDCG: {np.mean(xgb_scores_folds):.4f}")
#     print(f"LightGBM Average NDCG: {np.mean(lgbm_scores_folds):.4f}")
#     print(f"CatBoost Average NDCG: {np.mean(catboost_scores_folds):.4f}")
#     print(f"**Trinity Ensemble Average NDCG: {np.mean(ensemble_scores_folds):.4f}**")
