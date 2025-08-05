import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import numpy as np
from xgboost import XGBRegressor
import fastf1
from datetime import datetime
import pytz # Import pytz for timezone handling

# --- Configuration ---
HISTORICAL_DATA_FILE = "f1_historical_data_with_features.csv"
PREDICTION_CACHE_DIR = 'cache_prediction'
WINDOW_SIZE = 3 # For lagged features (average over last N races)

# --- Helper Functions (Copied from data_collection for prediction block needs) ---

def setup_cache(cache_dir):
    """Creates cache directory if it doesn't exist and enables FastF1 cache."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")
    fastf1.Cache.enable_cache(cache_dir)

def load_session_data_for_prediction(year, round_num, session_type):
    """Loads data for a specific F1 session using round number for prediction purposes."""
    print(f"Loading {session_type} session for {year} Round {round_num} for prediction...")
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(telemetry=False, laps=False, weather=False) # Only need basic info for drivers
        print(f"Successfully loaded {session_type} session for {year} Round {round_num} for prediction.")
        return session
    except Exception as e:
        print(f"Error loading session {year} Round {round_num} {session_type} for prediction: {e}")
        return None

# --- Main Model Training and Prediction Logic ---

if __name__ == "__main__":
    if not os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Error: Historical data file '{HISTORICAL_DATA_FILE}' not found.")
        print("Please run data_collection.py first to generate the data.")
        exit() # Correctly use exit() here

    print(f"Loading data from {HISTORICAL_DATA_FILE}...")
    df = pd.read_csv(HISTORICAL_DATA_FILE)

    # Ensure EventDate is in datetime format for sorting
    df['EventDate'] = pd.to_datetime(df['EventDate'], utc=True)

    # Define features and target
    numerical_features = [
        'QualiPosition',
        'AvgLapTime',
        'LapsCompleted',
        'AirTemp',
        'TrackTemp',
        'Rainfall',
        'DriverAvgQualiPositionLast3Races',
        'DriverAvgRaceFinishPositionLast3Races',
        'TeamAvgQualiPositionLast3Races',
        'TeamAvgRaceFinishPositionLast3Races',
        'TrackAvgQualiPosition', # New Feature
        'TrackAvgRaceFinishPosition' # New Feature
    ]
    categorical_features = ['Driver', 'Team'] # FullName is for display, not model input

    target = 'RaceFinishPosition'

    # Drop rows where the target or any critical numerical feature is missing
    df.dropna(subset=[target] + numerical_features, inplace=True)

    if df.empty:
        print("No valid data for training after dropping missing target or essential feature values. Cannot train model.")
        exit() # Correctly use exit() here
    else:
        # Sort data chronologically for correct time-based split
        df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)

        X = df[numerical_features + categorical_features]
        y = df[target]

        print(f"Features used (numerical): {numerical_features}")
        print(f"Features used (categorical): {categorical_features}")
        print(f"Target used: {target}")
        print(f"Dataset shape: {X.shape}")

        # --- Preprocessing Pipeline ---
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # Create a pipeline with preprocessing and the XGBoost Regressor model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', XGBRegressor(random_state=42))])

        # --- Hyperparameter Grid for GridSearchCV ---
        # Expanded grid for more thorough tuning
        param_grid = {
            'regressor__n_estimators': [100, 200, 300], # More trees
            'regressor__learning_rate': [0.01, 0.05, 0.1], # Wider range of learning rates
            'regressor__max_depth': [3, 5, 7], # Deeper trees
            'regressor__subsample': [0.7, 0.8, 1.0], # Wider range for sample ratio of training instances
            'regressor__colsample_bytree': [0.7, 0.8, 1.0], # Wider range for sample ratio of columns
            'regressor__gamma': [0, 0.1, 0.2], # Min loss reduction to make a split
            'regressor__min_child_weight': [1, 5, 10] # Min sum of instance weight needed in a child
        }

        # --- GridSearchCV Setup ---
        print("\nStarting GridSearchCV for hyperparameter tuning (this may take a while)...")
        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   cv=3,
                                   scoring='neg_mean_absolute_error',
                                   n_jobs=-1,
                                   verbose=2)

        # Time-based split: Train on 80% oldest data, test on 20% newest data
        split_index = int(len(df) * 0.8)
        X_train_df, X_test_df = df.iloc[:split_index], df.iloc[split_index:]
        y_train, y_test = df[target].iloc[:split_index], df[target].iloc[split_index:]

        X_train_features = X_train_df[numerical_features + categorical_features]
        X_test_features = X_test_df[numerical_features + categorical_features]

        print(f"Training data size for GridSearch: {len(X_train_features)}")
        print(f"Testing data size for final evaluation: {len(X_test_features)}")

        # Fit GridSearchCV on the training data
        grid_search.fit(X_train_features, y_train)
        print("GridSearchCV tuning complete.")

        # Get the best model found by GridSearchCV
        best_model_pipeline = grid_search.best_estimator_
        print(f"\nBest hyperparameters found: {grid_search.best_params_}")
        print(f"Best cross-validation MAE (negative): {grid_search.best_score_:.2f}")

        # Make predictions on the test set using the best model
        y_pred = best_model_pipeline.predict(X_test_features)

        # Evaluate the best model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n--- Final Model Evaluation (Best XGBoost from GridSearchCV) ---")
        print(f"Mean Absolute Error (MAE) on test set: {mae:.2f}")
        print(f"R-squared (R2) Score on test set: {r2:.2f}")
        print("\nSample Predictions vs Actual (from test set):")
        results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test}).reset_index(drop=True)
        print(results.head(10))

        # --- Dynamic Prediction for the Next Upcoming Race ---
        print("\n--- Predicting Lineup for the Next Upcoming Race ---")
        try:
            current_year = datetime.now().year
            setup_cache(PREDICTION_CACHE_DIR)

            schedule = fastf1.get_event_schedule(current_year)

            if schedule['EventDate'].dt.tz is None:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc)
            else:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_convert(pytz.utc)

            now_utc = pd.Timestamp.now(tz='UTC')
            upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')

            if upcoming_races.empty:
                print("No upcoming races found in the schedule for the current year.")
                print("Cannot perform dynamic prediction for the next race.")
            else:
                next_race_event = upcoming_races.iloc[0]
                next_race_year = current_year
                next_race_round = next_race_event['RoundNumber']
                next_race_name = next_race_event['EventName']

                print(f"\nAttempting to predict for: {next_race_year} {next_race_name} (Round {next_race_round})")

                drivers_for_prediction_list = []
                teams_for_prediction_list = []
                full_names_for_prediction_list = []

                # Attempt to get drivers from the upcoming qualifying session
                quali_session_next_race = load_session_data_for_prediction(next_race_year, next_race_round, 'Q')
                if quali_session_next_race and quali_session_next_race.drivers:
                    for d_num in quali_session_next_race.drivers:
                        driver_info = quali_session_next_race.get_driver(d_num)
                        drivers_for_prediction_list.append(driver_info['Abbreviation'])
                        teams_for_prediction_list.append(driver_info['TeamName'])
                        full_names_for_prediction_list.append(f"{driver_info['FirstName']} {driver_info['LastName']}")
                    # Sort lists consistently
                    drivers_for_prediction_list, teams_for_prediction_list, full_names_for_prediction_list = \
                        zip(*sorted(zip(drivers_for_prediction_list, teams_for_prediction_list, full_names_for_prediction_list)))
                    drivers_for_prediction_list = list(drivers_for_prediction_list)
                    teams_for_prediction_list = list(teams_for_prediction_list)
                    full_names_for_prediction_list = list(full_names_for_prediction_list)
                else:
                    print(f"Warning: Could not load qualifying session for {next_race_name} to get actual drivers.")
                    print("Using drivers from the last race in the training set for prediction input as a fallback.")
                    if not X_train_df.empty:
                        last_training_race_drivers_df = X_train_df[X_train_df['EventDate'] == X_train_df['EventDate'].max()]
                        if not last_training_race_drivers_df.empty:
                            drivers_for_prediction_list = last_training_race_drivers_df['Driver'].unique().tolist()
                            # Create a mapping from driver abbreviation to full name and team from the full df
                            driver_details_map = df[df['Driver'].isin(drivers_for_prediction_list)][['Driver', 'Team', 'FullName']].drop_duplicates().set_index('Driver')

                            # Use .get() with a default value for safety
                            teams_for_prediction_list = [driver_details_map['Team'].get(d, "Unknown") for d in drivers_for_prediction_list]
                            full_names_for_prediction_list = [driver_details_map['FullName'].get(d, d) for d in drivers_for_prediction_list]
                            drivers_for_prediction_list = sorted(list(drivers_for_prediction_list)) # Ensure consistent order
                        else:
                            print("Fallback to last training race drivers failed: Last training race data is empty.")
                    else:
                        print("Fallback to last training race drivers failed: Training data is empty.")

                if not drivers_for_prediction_list:
                    print("Could not determine drivers for prediction. Skipping dynamic prediction.")
                    exit()

                # Calculate lagged features based on data *before* this next race
                historical_data_for_prediction = df[df['EventDate'] < next_race_event['EventDate']].copy()

                # Calculate track-specific averages for the upcoming track from historical data
                track_avg_quali = historical_data_for_prediction[historical_data_for_prediction['GrandPrix'] == next_race_name]['QualiPosition'].mean()
                track_avg_race = historical_data_for_prediction[historical_data_for_prediction['GrandPrix'] == next_race_name]['RaceFinishPosition'].mean()

                if pd.isna(track_avg_quali):
                    track_avg_quali = df['TrackAvgQualiPosition'].mean()
                if pd.isna(track_avg_race):
                    track_avg_race = df['TrackAvgRaceFinishPosition'].mean()


                if historical_data_for_prediction.empty:
                    print("No historical data available before the next upcoming race to calculate lagged features.")
                    print("Using overall means from the full dataset for lagged features as a fallback.")
                    latest_driver_performance = pd.DataFrame({
                        'Driver': drivers_for_prediction_list,
                        'Team': teams_for_prediction_list,
                        'DriverAvgQualiPositionLast3Races': [df['DriverAvgQualiPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                        'DriverAvgRaceFinishPositionLast3Races': [df['DriverAvgRaceFinishPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                        'TeamAvgQualiPositionLast3Races': [df['TeamAvgQualiPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                        'TeamAvgRaceFinishPositionLast3Races': [df['TeamAvgRaceFinishPositionLast3Races'].mean()] * len(drivers_for_prediction_list)
                    })
                else:
                    historical_data_for_prediction.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)

                    historical_data_for_prediction['DriverAvgQualiPositionLast3Races'] = historical_data_for_prediction.groupby('Driver')['QualiPosition'].transform(
                        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
                    )
                    historical_data_for_prediction['DriverAvgRaceFinishPositionLast3Races'] = historical_data_for_prediction.groupby('Driver')['RaceFinishPosition'].transform(
                        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
                    )
                    historical_data_for_prediction['TeamAvgQualiPositionLast3Races'] = historical_data_for_prediction.groupby('Team')['QualiPosition'].transform(
                        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
                    )
                    historical_data_for_prediction['TeamAvgRaceFinishPositionLast3Races'] = historical_data_for_prediction.groupby('Team')['RaceFinishPosition'].transform(
                        lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
                    )

                    latest_driver_performance = historical_data_for_prediction.groupby('Driver').last().reset_index()
                    latest_team_performance = historical_data_for_prediction.groupby('Team').last().reset_index()

                    for col in ['DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
                                'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races']:
                        if latest_driver_performance[col].isnull().any():
                            latest_driver_performance[col].fillna(df[col].mean(), inplace=True)
                        if latest_team_performance[col].isnull().any():
                            latest_team_performance[col].fillna(df[col].mean(), inplace=True)


                # Prepare the input data for prediction
                prediction_data_rows = []
                for i, driver_code in enumerate(drivers_for_prediction_list):
                    team_name = teams_for_prediction_list[i] if i < len(teams_for_prediction_list) else "Unknown"
                    full_name = full_names_for_prediction_list[i] if i < len(full_names_for_prediction_list) else driver_code

                    driver_hist = latest_driver_performance[latest_driver_performance['Driver'] == driver_code]
                    team_hist = latest_team_performance[latest_team_performance['Team'] == team_name]

                    avg_lap_time = df['AvgLapTime'].mean()
                    laps_completed = df['LapsCompleted'].mean()
                    air_temp = df['AirTemp'].mean()
                    track_temp = df['TrackTemp'].mean()
                    rainfall = 0

                    driver_avg_quali = driver_hist['DriverAvgQualiPositionLast3Races'].iloc[0] if not driver_hist.empty else df['DriverAvgQualiPositionLast3Races'].mean()
                    driver_avg_race = driver_hist['DriverAvgRaceFinishPositionLast3Races'].iloc[0] if not driver_hist.empty else df['DriverAvgRaceFinishPositionLast3Races'].mean()
                    team_avg_quali = team_hist['TeamAvgQualiPositionLast3Races'].iloc[0] if not team_hist.empty else df['TeamAvgQualiPositionLast3Races'].mean()
                    team_avg_race = team_hist['TeamAvgRaceFinishPositionLast3Races'].iloc[0] if not team_hist.empty else df['TeamAvgRaceFinishPositionLast3Races'].mean()

                    prediction_data_rows.append({
                        'Driver': driver_code,
                        'FullName': full_name,
                        'Team': team_name,
                        'QualiPosition': i + 1,
                        'AvgLapTime': avg_lap_time,
                        'LapsCompleted': laps_completed,
                        'AirTemp': air_temp,
                        'TrackTemp': track_temp,
                        'Rainfall': rainfall,
                        'DriverAvgQualiPositionLast3Races': driver_avg_quali,
                        'DriverAvgRaceFinishPositionLast3Races': driver_avg_race,
                        'TeamAvgQualiPositionLast3Races': team_avg_quali,
                        'TeamAvgRaceFinishPositionLast3Races': team_avg_race,
                        'TrackAvgQualiPosition': track_avg_quali,
                        'TrackAvgRaceFinishPosition': track_avg_race
                    })

                prediction_input_df = pd.DataFrame(prediction_data_rows)

                next_race_predictions_raw = best_model_pipeline.predict(prediction_input_df[numerical_features + categorical_features])

                predicted_lineup_df = prediction_input_df.copy()
                predicted_lineup_df['PredictedFinishPosition'] = next_race_predictions_raw
                predicted_lineup_df = predicted_lineup_df.sort_values(by='PredictedFinishPosition').reset_index(drop=True)
                predicted_lineup_df['PredictedRank'] = predicted_lineup_df.index + 1

                print(predicted_lineup_df[['PredictedRank', 'FullName', 'PredictedFinishPosition']].head(20))

        except Exception as e:
            print(f"Error during dynamic next race prediction: {e}")
            print("Please ensure you have collected enough data and FastF1 can access upcoming schedules.")

        # You can also save the trained model for later use
        # import joblib
        # joblib.dump(best_model_pipeline, 'f1_winner_predictor_tuned_pipeline.pkl')
        # print("\nModel saved as f1_winner_predictor_tuned_pipeline.pkl")
