import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRanker
import fastf1
from datetime import datetime
import pytz

# --- Configuration ---
HISTORICAL_DATA_FILE = "f1_historical_data_with_features.csv"
PREDICTION_CACHE_DIR = 'cache_prediction'


# --- Main Model Training and Prediction Logic ---
if __name__ == "__main__":
    if not os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Error: '{HISTORICAL_DATA_FILE}' not found. Run data_collection.py first.")
        exit()

    print(f"Loading data from {HISTORICAL_DATA_FILE}...")
    df = pd.read_csv(HISTORICAL_DATA_FILE)
    df['EventDate'] = pd.to_datetime(df['EventDate'], utc=True)

    # --- Feature Lists ---
    numerical_features = [
        'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
        'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
        'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
        'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
        'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
        'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
        'PracticeFastestLap', 'AvgLongRunPace'
    ]
    categorical_features = ['Driver', 'Team', 'StartingTyreCompound']
    target = 'RaceFinishPosition'

    df.dropna(subset=[target] + numerical_features, inplace=True)
    df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
    
    # Invert target for ranking: higher score = better rank
    df[target] = df.groupby(['Year', 'RoundNumber'])[target].transform(lambda x: x.max() - x + 1)
    
    # Create a unique Group ID for each race for GroupKFold
    df['group_id'] = df.groupby(['Year', 'RoundNumber']).ngroup()

    X_full = df[numerical_features + categorical_features]
    y_full = df[target]
    groups_full = df.groupby(['Year', 'RoundNumber']).size().to_numpy()
    group_ids_full = df['group_id']
    
    print(f"Data prepared for ranking with {len(groups_full)} race groups.")

    # --- Preprocessing Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='drop')

    # --- Hyperparameter Grid ---
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    # --- FULL HYPERPARAMETER TUNING LOOP WITH GROUPKFOLD ---
    print("\n--- Performing robust hyperparameter tuning with GroupKFold ---")

    gkf = GroupKFold(n_splits=3)
    best_score = -np.inf
    best_params = {}

    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

    for params in param_combinations:
        current_scores = []
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('ranker', XGBRanker(objective='rank:ndcg', **params, random_state=42))])

        for train_idx, val_idx in gkf.split(X_full, y_full, groups=group_ids_full):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
            
            train_groups = df.iloc[train_idx].groupby('group_id').size().to_numpy()
            
            pipeline.fit(X_train, y_train, ranker__group=train_groups)
            
            y_pred_scores = pipeline.predict(X_val)
            
            val_df = pd.DataFrame({'y_val': y_val, 'y_pred_scores': y_pred_scores, 'group_id': group_ids_full.iloc[val_idx]})
            
            group_ndcgs = []
            for group_id in val_df['group_id'].unique():
                group_data = val_df[val_df['group_id'] == group_id]
                if len(group_data) > 1:
                    ndcg = ndcg_score([group_data['y_val'].values], [group_data['y_pred_scores'].values])
                    group_ndcgs.append(ndcg)
            
            if group_ndcgs:
                current_scores.append(np.mean(group_ndcgs))

        avg_score = np.mean(current_scores) if current_scores else -1
        print(f"Params: {params} --> Avg NDCG Score: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print(f"\nBest hyperparameters found: {best_params} with NDCG score: {best_score:.4f}")

    # --- Final Model Training on Entire Dataset ---
    print("\n--- Training final XGBRanker model with best parameters ---")
    best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                          ('ranker', XGBRanker(objective='rank:ndcg', **best_params, random_state=42))])
    
    best_model_pipeline.fit(X_full, y_full, ranker__group=groups_full)
    print("Final model training complete.")

    # --- DYNAMIC PREDICTION FOR NEXT UPCOMING RACE ---
    try:
        if not os.path.exists(PREDICTION_CACHE_DIR):
            os.makedirs(PREDICTION_CACHE_DIR)
        fastf1.Cache.enable_cache(PREDICTION_CACHE_DIR)
        
        current_year = datetime.now().year
        schedule = fastf1.get_event_schedule(current_year)
        
        # Ensure consistent UTC timezone for proper date comparison
        schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc) if schedule['EventDate'].dt.tz is None else schedule['EventDate'].dt.tz_convert(pytz.utc)
        now_utc = pd.Timestamp.now(tz='UTC')
        
        upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')

        if upcoming_races.empty:
            print("\nNo upcoming races found for the current year.")
        else:
            next_race_event = upcoming_races.iloc[0]
            next_race_name = next_race_event['EventName']
            
            print(f"\n--- Predicting Lineup for: {current_year} {next_race_name} ---")
            print("(Note: Using feature values from the last known race as a template for prediction)")
            
            # Use the last race from our dataset as a template for the prediction input
            last_race_mask = df['group_id'] == df['group_id'].max()
            prediction_input_df = df[last_race_mask].copy()

            if prediction_input_df.empty:
                 print("Could not create prediction input from the last race.")
            else:
                predicted_scores = best_model_pipeline.predict(prediction_input_df[numerical_features + categorical_features])
                prediction_input_df['PredictedScore'] = predicted_scores
                
                predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
                predicted_lineup_df['PredictedRank'] = predicted_lineup_df.index + 1
                
                print(predicted_lineup_df[['PredictedRank', 'FullName', 'Team', 'PredictedScore']])

    except Exception as e:
        print(f"\nCould not perform dynamic prediction for the next race. Error: {e}")
        print("Displaying prediction using last known race as a fallback.")
