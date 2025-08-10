#TRINITY

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
from lightgbm import LGBMRanker
from catboost import CatBoostRanker
import joblib
import sqlite3
import warnings

# Suppress the specific UserWarning from scikit-learn about feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but LGBMRanker was fitted with feature names")


# --- Configuration ---
DB_FILE = "f1_predictor.db"
XGB_MODEL_FILE = "Trinity/xgb_model.joblib"
LGBM_MODEL_FILE = "Trinity/lgbm_model.joblib"
CATBOOST_MODEL_FILE = "Trinity/catboost_model.joblib"

# --- Main Model Training Logic ---
if __name__ == "__main__":
    # Create the model directory if it doesn't exist
    os.makedirs("Trinity", exist_ok=True)

    if not os.path.exists(DB_FILE):
        print(f"Error: '{DB_FILE}' not found. Run data_collection.py first.")
        exit()

    print(f"Loading data from {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql('SELECT * FROM historical_data', conn)
    conn.close()
    
    df['EventDate'] = pd.to_datetime(df['EventDate'], utc=True)

    # --- Feature Lists ---
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
    target = 'RaceFinishPosition'

    df.dropna(subset=[target] + numerical_features, inplace=True)
    df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
    
    df[target] = df.groupby(['Year', 'RoundNumber'])[target].transform(lambda x: x.max() - x + 1)
    df['group_id'] = df.groupby(['Year', 'RoundNumber']).ngroup()

    X_full = df[numerical_features + categorical_features]
    y_full = df[target]
    groups_full = df.groupby(['Year', 'RoundNumber']).size().to_numpy()
    group_ids_full = df['group_id']
    
    print(f"Data prepared for ranking with {len(groups_full)} race groups.")

    # --- Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='drop')

    # --- Model Definitions ---
    xgb_params = {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'gamma': 0.1, 'min_child_weight': 1}
    lgbm_params = {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8}
    catboost_params = {'iterations': 300, 'learning_rate': 0.1, 'depth': 3, 'verbose': 0}

    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', XGBRanker(objective='rank:ndcg', **xgb_params, random_state=42))])
    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', LGBMRanker(objective='lambdarank', **lgbm_params, random_state=42))])
    catboost_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('ranker', CatBoostRanker(objective='QueryRMSE', **catboost_params, random_state=42))])

    # --- Ensemble Evaluation with GroupKFold ---
    print("\n--- Evaluating Ensemble Performance with GroupKFold Cross-Validation ---")
    gkf = GroupKFold(n_splits=3)
    
    xgb_scores_folds, lgbm_scores_folds, catboost_scores_folds, ensemble_scores_folds = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y_full, groups=group_ids_full)):
        print(f"\n--- Processing Fold {fold+1}/3 ---")
        X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        train_groups = df.iloc[train_idx].groupby('group_id').size().to_numpy()
        
        # Train and evaluate each model individually
        print("Training and Evaluating XGBoost...")
        xgb_pipeline.fit(X_train, y_train, ranker__group=train_groups)
        xgb_pred = xgb_pipeline.predict(X_val)
        
        print("Training and Evaluating LightGBM...")
        lgbm_pipeline.fit(X_train, y_train, ranker__group=train_groups)
        lgbm_pred = lgbm_pipeline.predict(X_val)
        
        print("Training and Evaluating CatBoost...")
        X_train_cat, X_val_cat = X_train.copy(), X_val.copy()
        X_train_cat['group_id'] = df.iloc[train_idx]['group_id']
        catboost_pipeline.fit(X_train_cat, y_train, ranker__group_id=X_train_cat['group_id'])
        catboost_pred = catboost_pipeline.predict(X_val_cat)
        
        # Evaluate individual models and the ensemble
        val_df = pd.DataFrame({'y_val': y_val, 'group_id': group_ids_full.iloc[val_idx],
                               'xgb_pred': xgb_pred, 'lgbm_pred': lgbm_pred, 'catboost_pred': catboost_pred})
        val_df['ensemble_pred'] = (val_df['xgb_pred'] + val_df['lgbm_pred'] + val_df['catboost_pred']) / 3.0

        fold_xgb_scores, fold_lgbm_scores, fold_catboost_scores, fold_ensemble_scores = [], [], [], []
        for group_id in val_df['group_id'].unique():
            group_data = val_df[val_df['group_id'] == group_id]
            if len(group_data) > 1:
                fold_xgb_scores.append(ndcg_score([group_data['y_val'].values], [group_data['xgb_pred'].values]))
                fold_lgbm_scores.append(ndcg_score([group_data['y_val'].values], [group_data['lgbm_pred'].values]))
                fold_catboost_scores.append(ndcg_score([group_data['y_val'].values], [group_data['catboost_pred'].values]))
                fold_ensemble_scores.append(ndcg_score([group_data['y_val'].values], [group_data['ensemble_pred'].values]))
        
        xgb_scores_folds.append(np.mean(fold_xgb_scores))
        lgbm_scores_folds.append(np.mean(fold_lgbm_scores))
        catboost_scores_folds.append(np.mean(fold_catboost_scores))
        ensemble_scores_folds.append(np.mean(fold_ensemble_scores))

    # --- Train and Save Final Models on ALL Data ---
    print("\n--- Training Final Models on ALL Data for Production ---")
    print("Training XGBoost...")
    xgb_pipeline.fit(X_full, y_full, ranker__group=groups_full)
    joblib.dump(xgb_pipeline, XGB_MODEL_FILE)
    print(f"XGBoost model saved to {XGB_MODEL_FILE}")

    print("Training LightGBM...")
    lgbm_pipeline.fit(X_full, y_full, ranker__group=groups_full)
    joblib.dump(lgbm_pipeline, LGBM_MODEL_FILE)
    print(f"LightGBM model saved to {LGBM_MODEL_FILE}")

    print("Training CatBoost...")
    X_full_cat = X_full.copy()
    X_full_cat['group_id'] = df['group_id']
    catboost_pipeline.fit(X_full_cat, y_full, ranker__group_id=X_full_cat['group_id'])
    joblib.dump(catboost_pipeline, CATBOOST_MODEL_FILE)
    print(f"CatBoost model saved to {CATBOOST_MODEL_FILE}")

    print("\n--- All models trained and saved successfully! ---")

    # --- Final Score Summary ---
    print("\n--- Cross-Validation NDCG Scores ---")
    print(f"XGBoost Scores per Fold: {[round(s, 4) for s in xgb_scores_folds]}")
    print(f"LightGBM Scores per Fold: {[round(s, 4) for s in lgbm_scores_folds]}")
    print(f"CatBoost Scores per Fold: {[round(s, 4) for s in catboost_scores_folds]}")
    print(f"Trinity Ensemble Scores per Fold: {[round(s, 4) for s in ensemble_scores_folds]}")
    print("\n--- Average NDCG Scores ---")
    print(f"XGBoost Average NDCG: {np.mean(xgb_scores_folds):.4f}")
    print(f"LightGBM Average NDCG: {np.mean(lgbm_scores_folds):.4f}")
    print(f"CatBoost Average NDCG: {np.mean(catboost_scores_folds):.4f}")
    print(f"**Trinity Ensemble Average NDCG: {np.mean(ensemble_scores_folds):.4f}**")






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


# # --- Configuration ---
# DB_FILE = "f1_predictor.db"
# XGB_MODEL_FILE = "Trinity/xgb_model.joblib"
# LGBM_MODEL_FILE = "Trinity/lgbm_model.joblib"
# CATBOOST_MODEL_FILE = "Trinity/catboost_model.joblib"

# # --- Main Model Training Logic ---
# if __name__ == "__main__":
#     # --- FIX: Create the model directory if it doesn't exist ---
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

#         xgb_fold_scores, lgbm_fold_scores, catboost_fold_scores, ensemble_fold_scores = [], [], [], []
#         for group_id in val_df['group_id'].unique():
#             group_data = val_df[val_df['group_id'] == group_id]
#             if len(group_data) > 1:
#                 xgb_fold_scores.append(ndcg_score([group_data['y_val'].values], [group_data['xgb_pred'].values]))
#                 lgbm_fold_scores.append(ndcg_score([group_data['y_val'].values], [group_data['lgbm_pred'].values]))
#                 catboost_fold_scores.append(ndcg_score([group_data['y_val'].values], [group_data['catboost_pred'].values]))
#                 ensemble_fold_scores.append(ndcg_score([group_data['y_val'].values], [group_data['ensemble_pred'].values]))
        
#         xgb_scores_folds.append(np.mean(xgb_fold_scores))
#         lgbm_scores_folds.append(np.mean(lgbm_fold_scores))
#         catboost_scores_folds.append(np.mean(catboost_fold_scores))
#         ensemble_scores_folds.append(np.mean(ensemble_fold_scores))

#     # --- Final Score Summary ---
#     print("\n\n--- All models trained and saved successfully! ---")
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


# # --- Configuration ---
# DB_FILE = "f1_predictor.db" # UPDATED: Database file name
# XGB_MODEL_FILE = "xgb_model.joblib"
# LGBM_MODEL_FILE = "lgbm_model.joblib"
# CATBOOST_MODEL_FILE = "catboost_model.joblib"

# # --- Main Model Training Logic ---
# if __name__ == "__main__":
#     if not os.path.exists(DB_FILE):
#         print(f"Error: '{DB_FILE}' not found. Run data_collection.py first.")
#         exit()

#     print(f"Loading data from {DB_FILE}...")
#     # --- UPDATED: Read from SQLite Database ---
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

#     # --- Train and Save Each Model ---
#     print("\n--- Training XGBoost Model ---")
#     xgb_pipeline.fit(X_full, y_full, ranker__group=groups_full)
#     joblib.dump(xgb_pipeline, XGB_MODEL_FILE)
#     print(f"XGBoost model saved to {XGB_MODEL_FILE}")

#     print("\n--- Training LightGBM Model ---")
#     lgbm_pipeline.fit(X_full, y_full, ranker__group=groups_full)
#     joblib.dump(lgbm_pipeline, LGBM_MODEL_FILE)
#     print(f"LightGBM model saved to {LGBM_MODEL_FILE}")

#     print("\n--- Training CatBoost Model ---")
#     X_full_cat = X_full.copy()
#     X_full_cat['group_id'] = df['group_id']
#     catboost_pipeline.fit(X_full_cat, y_full, ranker__group_id=X_full_cat['group_id'])
#     joblib.dump(catboost_pipeline, CATBOOST_MODEL_FILE)
#     print(f"CatBoost model saved to {CATBOOST_MODEL_FILE}")

#     print("\n--- All models trained and saved successfully! ---")
