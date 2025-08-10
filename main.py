#JAI SHREE RAM //
#JAI MATA DI //
#JAI HANUMAN JI //
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import fastf1
import sqlite3
import json

def run_data_collection():
    print("Running data collection...")
    import fastf1
import pandas as pd
import os
from datetime import datetime
import pytz 
import numpy as np 
from fastf1.core import DataNotLoadedError
import sqlite3

# --- Configuration ---
START_YEAR = 2018
DB_FILE = "f1_predictor.db" 
MAIN_CACHE_DIR = 'cache_main'
WINDOW_SIZE = 3 

# --- Static Track Characteristics ---
TRACK_CHARACTERISTICS = {
    'Bahrain Grand Prix': {'LengthKm': 5.412, 'NumCorners': 15}, 'Saudi Arabian Grand Prix': {'LengthKm': 6.174, 'NumCorners': 27},
    'Australian Grand Prix': {'LengthKm': 5.278, 'NumCorners': 14}, 'Japanese Grand Prix': {'LengthKm': 5.807, 'NumCorners': 18},
    'Chinese Grand Prix': {'LengthKm': 5.451, 'NumCorners': 16}, 'Miami Grand Prix': {'LengthKm': 5.412, 'NumCorners': 19},
    'Emilia Romagna Grand Prix': {'LengthKm': 4.909, 'NumCorners': 19}, 'Monaco Grand Prix': {'LengthKm': 3.337, 'NumCorners': 19},
    'Spanish Grand Prix': {'LengthKm': 4.657, 'NumCorners': 14}, 'Canadian Grand Prix': {'LengthKm': 4.361, 'NumCorners': 14},
    'Austrian Grand Prix': {'LengthKm': 4.318, 'NumCorners': 10}, 'British Grand Prix': {'LengthKm': 5.891, 'NumCorners': 18},
    'Hungarian Grand Prix': {'LengthKm': 4.381, 'NumCorners': 14}, 'Belgian Grand Prix': {'LengthKm': 7.004, 'NumCorners': 19},
    'Dutch Grand Prix': {'LengthKm': 4.259, 'NumCorners': 14}, 'Italian Grand Prix': {'LengthKm': 5.793, 'NumCorners': 11},
    'Azerbaijan Grand Prix': {'LengthKm': 6.003, 'NumCorners': 20}, 'Singapore Grand Prix': {'LengthKm': 4.940, 'NumCorners': 19},
    'United States Grand Prix': {'LengthKm': 5.513, 'NumCorners': 20}, 'Mexico City Grand Prix': {'LengthKm': 4.304, 'NumCorners': 17},
    'Brazilian Grand Prix': {'LengthKm': 4.309, 'NumCorners': 15}, 'Las Vegas Grand Prix': {'LengthKm': 6.201, 'NumCorners': 17},
    'Abu Dhabi Grand Prix': {'LengthKm': 5.281, 'NumCorners': 16}, 'Qatar Grand Prix': {'LengthKm': 5.419, 'NumCorners': 16},
    'French Grand Prix': {'LengthKm': 5.842, 'NumCorners': 15}, 'Styrian Grand Prix': {'LengthKm': 4.318, 'NumCorners': 10},
    'Tuscan Grand Prix': {'LengthKm': 5.245, 'NumCorners': 15}, 'Eifel Grand Prix': {'LengthKm': 5.148, 'NumCorners': 15},
    'Portuguese Grand Prix': {'LengthKm': 4.653, 'NumCorners': 15}, 'Turkish Grand Prix': {'LengthKm': 5.338, 'NumCorners': 14},
    'German Grand Prix': {'LengthKm': 4.574, 'NumCorners': 17}, 'Russian Grand Prix': {'LengthKm': 5.848, 'NumCorners': 18},
}

# --- Track Type Classification ---
TRACK_TYPES = {
    'Bahrain Grand Prix': 'Balanced', 'Saudi Arabian Grand Prix': 'Street Circuit', 'Australian Grand Prix': 'Street Circuit',
    'Japanese Grand Prix': 'High Downforce', 'Chinese Grand Prix': 'Balanced', 'Miami Grand Prix': 'Street Circuit',
    'Emilia Romagna Grand Prix': 'High Downforce', 'Monaco Grand Prix': 'Street Circuit', 'Spanish Grand Prix': 'High Downforce',
    'Canadian Grand Prix': 'Low Downforce', 'Austrian Grand Prix': 'Low Downforce', 'British Grand Prix': 'Low Downforce',
    'Hungarian Grand Prix': 'High Downforce', 'Belgian Grand Prix': 'Low Downforce', 'Dutch Grand Prix': 'High Downforce',
    'Italian Grand Prix': 'Low Downforce', 'Azerbaijan Grand Prix': 'Street Circuit', 'Singapore Grand Prix': 'Street Circuit',
    'United States Grand Prix': 'Balanced', 'Mexico City Grand Prix': 'High Altitude', 'Brazilian Grand Prix': 'High Altitude',
    'Las Vegas Grand Prix': 'Street Circuit', 'Abu Dhabi Grand Prix': 'Balanced', 'Qatar Grand Prix': 'Balanced',
    'French Grand Prix': 'Balanced', 'Styrian Grand Prix': 'Low Downforce', 'Tuscan Grand Prix': 'High Downforce',
    'Eifel Grand Prix': 'Balanced', 'Portuguese Grand Prix': 'Balanced', 'Turkish Grand Prix': 'High Downforce',
    'German Grand Prix': 'Balanced', 'Russian Grand Prix': 'Street Circuit'
}

# --- Helper Functions ---
def setup_cache(cache_dir):
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

def load_session_data(year, round_num, session_type):
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(telemetry=False, laps=True, weather=True)
        return session
    except Exception as e:
        print(f"Could not load {session_type} for {year} Round {round_num}: {e}")
        return None

def get_driver_features(session):
    if session is None: return pd.DataFrame()
    driver_data = []
    for driver_num in session.drivers:
        try:
            driver_info = session.get_driver(driver_num)
            laps = session.laps.pick_drivers([driver_info['Abbreviation']])
            if not laps.empty:
                valid_laps = laps.loc[laps['IsAccurate']]
                weather = session.weather_data
                driver_data.append({
                    'Driver': driver_info['Abbreviation'], 'FullName': f"{driver_info['FirstName']} {driver_info['LastName']}", 'Team': driver_info['TeamName'],
                    'AvgLapTime': valid_laps['LapTime'].dt.total_seconds().median() if not valid_laps.empty else np.nan,
                    'LapsCompleted': len(laps), 'AirTemp': weather['AirTemp'].iloc[0] if not weather.empty else np.nan,
                    'TrackTemp': weather['TrackTemp'].iloc[0] if not weather.empty else np.nan,
                    'Rainfall': weather['Rainfall'].iloc[0] if not weather.empty else 0
                })
        except Exception as e:
            print(f"Error processing driver {driver_num}: {e}")
    return pd.DataFrame(driver_data)

def get_practice_features(year, round_num, race_drivers_list):
    all_practice_laps_list = []
    for fp in ['FP1', 'FP2', 'FP3']:
        session = load_session_data(year, round_num, fp)
        if session:
            try:
                laps = session.laps
                if not laps.empty:
                    all_practice_laps_list.append(laps)
            except DataNotLoadedError:
                print(f"Practice session {fp} for {year} Round {round_num} has no lap data. Skipping.")
                continue
    
    if not all_practice_laps_list:
        return pd.DataFrame()

    practice_laps = pd.concat(all_practice_laps_list)
    if practice_laps.empty: return pd.DataFrame()
    
    practice_features = []
    for driver in race_drivers_list:
        driver_laps = practice_laps.pick_drivers([driver])
        if driver_laps.empty: continue
        stints = driver_laps.groupby('Stint')
        long_run_paces = [s['LapTime'].median().total_seconds() for _, s in stints if len(s) >= 5]
        practice_features.append({
            'Driver': driver,
            'PracticeFastestLap': driver_laps['LapTime'].min().total_seconds() if not driver_laps.empty else np.nan,
            'AvgLongRunPace': min(long_run_paces) if long_run_paces else np.nan
        })
    return pd.DataFrame(practice_features)

# --- Main Data Collection Logic ---
if __name__ == "__main__":
    current_year = datetime.now().year
    years_to_collect = list(range(START_YEAR, current_year + 1)) 
    all_races_data = []
    setup_cache(MAIN_CACHE_DIR)
    for year in years_to_collect:
        print(f"\n--- Collecting data for {year} season ---")
        
        # --- Handle years with no schedule data ---
        try:
            schedule = fastf1.get_event_schedule(year)
        except ValueError:
            print(f"Could not load schedule for {year}. It may not be available yet. Skipping.")
            continue # Skip to the next year
            
        schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc) if schedule['EventDate'].dt.tz is None else schedule['EventDate'].dt.tz_convert(pytz.utc)
        
        # Filter to only include races that have already happened
        schedule = schedule[schedule['EventDate'] < pd.Timestamp.now(tz='UTC')]

        for index, event in schedule.iterrows():
            event_name, round_num = event['EventName'], event['RoundNumber']
            if "testing" in event_name.lower() or round_num == 0: continue
            print(f"\nProcessing {year} {event_name}...")
            quali_session = load_session_data(year, round_num, 'Q')
            if quali_session is None: continue
            
            quali_features_df = get_driver_features(quali_session)
            if quali_features_df.empty:
                print(f"No driver data extracted for qualifying in {year} {event_name}. Skipping race.")
                continue

            race_drivers_list = quali_features_df['Driver'].unique().tolist()
            practice_features_df = get_practice_features(year, round_num, race_drivers_list)
            
            if not quali_session.results.empty:
                quali_results = quali_session.results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'})
                quali_features_df = pd.merge(quali_features_df, quali_results, on='Driver', how='left')
            
            race_session = load_session_data(year, round_num, 'R')
            if race_session is None or race_session.results.empty: continue
            
            race_results = race_session.results[['Abbreviation', 'Position', 'TeamName', 'Status']].rename(columns={'Abbreviation': 'Driver', 'Position': 'RaceFinishPosition', 'TeamName': 'Team'})
            race_results['IsDNF'] = race_results['Status'].apply(lambda x: 0 if 'Lap' in x or 'Finished' in x else 1)
            
            teammate_battles = []
            for team in race_results['Team'].unique():
                team_drivers = race_results[race_results['Team'] == team]
                if len(team_drivers) == 2:
                    d1, d2 = team_drivers.iloc[0], team_drivers.iloc[1]
                    winner, loser = (d1, d2) if d1['RaceFinishPosition'] < d2['RaceFinishPosition'] else (d2, d1)
                    teammate_battles.append({'Driver': winner['Driver'], 'BeatTeammate': 1})
                    teammate_battles.append({'Driver': loser['Driver'], 'BeatTeammate': 0})
            if teammate_battles:
                race_results = pd.merge(race_results, pd.DataFrame(teammate_battles), on='Driver', how='left')

            merged_df = pd.merge(quali_features_df, race_results, on=['Driver', 'Team'], how='inner')
            if not practice_features_df.empty:
                merged_df = pd.merge(merged_df, practice_features_df, on='Driver', how='left')
            
            for col in ['NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace']:
                merged_df[col] = np.nan
            merged_df['StartingTyreCompound'] = 'UNKNOWN'

            try:
                if not race_session.laps.empty:
                    for driver_code in merged_df['Driver']:
                        laps = race_session.laps.pick_drivers([driver_code])
                        if not laps.empty:
                            accurate_laps = laps.loc[laps['IsAccurate']]
                            num_pits = len(laps.loc[laps['PitInTime'].notna()])
                            laps_completed = merged_df.loc[merged_df['Driver'] == driver_code, 'LapsCompleted'].iloc[0]
                            merged_df.loc[merged_df['Driver'] == driver_code, 'NumPitStops'] = num_pits
                            merged_df.loc[merged_df['Driver'] == driver_code, 'AvgStintLength'] = laps_completed / (num_pits + 1) if laps_completed > 0 else 0
                            merged_df.loc[merged_df['Driver'] == driver_code, 'StartingTyreCompound'] = laps.pick_track_status('1')['Compound'].iloc[0] if not laps.pick_track_status('1').empty else 'UNKNOWN'
                            if not accurate_laps.empty:
                                merged_df.loc[merged_df['Driver'] == driver_code, 'MedianLapTime'] = accurate_laps['LapTime'].dt.total_seconds().median()
                                merged_df.loc[merged_df['Driver'] == driver_code, 'StdDevLapTime'] = accurate_laps['LapTime'].dt.total_seconds().std()
                                merged_df.loc[merged_df['Driver'] == driver_code, 'AvgSector2Pace'] = accurate_laps['Sector2Time'].dt.total_seconds().mean()
            except DataNotLoadedError:
                 print(f"Race session for {year} {event_name} has no lap data. Skipping strategy features.")

            merged_df['Year'], merged_df['GrandPrix'], merged_df['EventDate'], merged_df['RoundNumber'] = year, event_name, event['EventDate'], round_num
            merged_df['TrackType'] = TRACK_TYPES.get(event_name, 'Unknown')
            track_info = TRACK_CHARACTERISTICS.get(event_name, {})
            merged_df['TrackLengthKm'] = track_info.get('LengthKm', np.nan)
            merged_df['NumCorners'] = track_info.get('NumCorners', np.nan)
            all_races_data.append(merged_df)

    if all_races_data:
        final_df = pd.concat(all_races_data, ignore_index=True)
        final_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
        
        # --- Engineer All Rolling/Expanding Features ---
        final_df['DriverAvgQualiPositionLast3Races'] = final_df.groupby('Driver')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['DriverAvgRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['DriverStdDevRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).std()).fillna(0)
        final_df['TeamAvgQualiPositionLast3Races'] = final_df.groupby('Team')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['TeamAvgRaceFinishPositionLast3Races'] = final_df.groupby('Team')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['TrackAvgQualiPosition'] = final_df.groupby('GrandPrix')['QualiPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        final_df['TrackAvgRaceFinishPosition'] = final_df.groupby('GrandPrix')['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        final_df['TeammateWinRate'] = final_df.groupby('Driver')['BeatTeammate'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        final_df['DriverDNFRate'] = final_df.groupby('Driver')['IsDNF'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        final_df['TrackTypePerformance'] = final_df.groupby(['Driver', 'TrackType'])['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        cols_to_fill = [
            'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races',
            'TrackAvgQualiPosition', 'TrackAvgRaceFinishPosition', 'NumPitStops', 'MedianLapTime',
            'StdDevLapTime', 'AvgSector2Pace', 'PracticeFastestLap', 'AvgLongRunPace',
            'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance', 'BeatTeammate',
            'TrackLengthKm', 'NumCorners', 'AvgStintLength'
        ]
        for col in cols_to_fill:
            if col in final_df.columns and final_df[col].isnull().any():
                final_df[col] = final_df[col].fillna(final_df[col].mean())
        
        conn = sqlite3.connect(DB_FILE)
        final_df.to_sql('historical_data', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\nConsolidated data with all new features saved to SQLite database: {DB_FILE}")

    print("Data collection completed.")

def run_model_training():
    print("Running model training...")
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

    print("Model training completed.")

def main():
    try:
        print("Starting the F1 predictor pipeline...")
        run_data_collection()
        run_model_training()
        print("Pipeline finished successfully. New model files are saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
