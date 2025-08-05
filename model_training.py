import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import numpy as np
from xgboost import XGBRegressor
import fastf1 # Import FastF1 for getting upcoming schedule
from datetime import datetime
import pytz # Import pytz for timezone handling

def train_and_evaluate_model(data_path):
    """
    Loads data, trains an XGBoost Regressor model with GridSearchCV, and evaluates it.
    Also performs a dynamic prediction for the next upcoming race.

    Args:
        data_path (str): Path to the CSV file containing the data.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run data_collection.py first.")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

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
        'TeamAvgRaceFinishPositionLast3Races'
    ]
    categorical_features = ['Driver', 'Team']

    target = 'RaceFinishPosition'

    # Drop rows where the target or any critical numerical feature is missing
    df.dropna(subset=[target] + numerical_features, inplace=True)

    if df.empty:
        print("No valid data for training after dropping missing target or essential feature values. Cannot train model.")
        return

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
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0]
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

    # X_train and X_test now contain the full dataframes with all columns
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
        # Get the current year's schedule
        current_year = datetime.now().year
        # Define prediction cache directory and create it if it doesn't exist
        PREDICTION_CACHE_DIR = 'cache_prediction'
        if not os.path.exists(PREDICTION_CACHE_DIR):
            os.makedirs(PREDICTION_CACHE_DIR)
            print(f"Created prediction cache directory: {PREDICTION_CACHE_DIR}")
        fastf1.Cache.enable_cache(PREDICTION_CACHE_DIR)

        schedule = fastf1.get_event_schedule(current_year)

        # Convert EventDate to timezone-aware UTC for consistent comparison
        if schedule['EventDate'].dt.tz is None:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc)
        else:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_convert(pytz.utc)

        # Find the next race (first race in schedule that hasn't occurred yet)
        now_utc = pd.Timestamp.now(tz='UTC')
        upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')

        if upcoming_races.empty:
            print("No upcoming races found in the schedule for the current year.")
            print("Cannot perform dynamic prediction for the next race.")
        else:
            next_race_event = upcoming_races.iloc[0]
            # Use the 'year' from the outer loop for robustness, as it's the current year being processed
            next_race_year = current_year
            next_race_round = next_race_event['RoundNumber']
            next_race_name = next_race_event['EventName']

            print(f"\nAttempting to predict for: {next_race_year} {next_race_name} (Round {next_race_round})")

            # Load the qualifying session for the next race to get actual drivers
            # This will likely fail if the session hasn't happened yet, but we need driver list
            quali_session_next_race = None
            drivers_for_prediction_list = []
            teams_for_prediction_list = []

            try:
                quali_session_next_race = fastf1.get_session(next_race_year, next_race_round, 'Q')
                quali_session_next_race.load(telemetry=False, laps=False, weather=False) # Only need driver info
                if quali_session_next_race.drivers:
                    drivers_for_prediction_list = [quali_session_next_race.get_driver(d)['Abbreviation'] for d in quali_session_next_race.drivers]
                    teams_for_prediction_list = [quali_session_next_race.get_driver(d)['TeamName'] for d in quali_session_next_race.drivers]
            except Exception as e:
                print(f"Warning: Could not load qualifying session for {next_race_name} to get actual drivers: {e}")
                print("Using drivers from the last race in the training set for prediction input as a fallback.")
                # Fallback: Use drivers from the last race in the training set
                # Ensure X_train_df is not empty before attempting to access its max EventDate
                if not X_train_df.empty:
                    last_training_race_drivers_df = X_train_df[X_train_df['EventDate'] == X_train_df['EventDate'].max()]
                    if not last_training_race_drivers_df.empty:
                        drivers_for_prediction_list = last_training_race_drivers_df['Driver'].unique().tolist()
                        driver_team_map = last_training_race_drivers_df[['Driver', 'Team']].drop_duplicates().set_index('Driver')('Team').to_dict()
                        teams_for_prediction_list = [driver_team_map.get(d) for d in drivers_for_prediction_list]
                        drivers_for_prediction_list = sorted(list(drivers_for_prediction_list)) # Ensure consistent order
                    else:
                        print("Fallback to last training race drivers failed: Last training race data is empty.")
                else:
                    print("Fallback to last training race drivers failed: Training data is empty.")

            if not drivers_for_prediction_list:
                print("Could not determine drivers for prediction. Skipping dynamic prediction.")
                return


            # Calculate lagged features based on data *before* this next race
            # This is the 'historical_data_for_prediction'
            historical_data_for_prediction = df[df['EventDate'] < next_race_event['EventDate']].copy()

            if historical_data_for_prediction.empty:
                print("No historical data available before the next upcoming race to calculate lagged features.")
                print("Using overall means from the full dataset for lagged features as a fallback.")
                # Fallback to overall means if no historical data before this race
                latest_driver_performance = pd.DataFrame({
                    'Driver': drivers_for_prediction_list,
                    'Team': teams_for_prediction_list,
                    'DriverAvgQualiPositionLast3Races': [df['DriverAvgQualiPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                    'DriverAvgRaceFinishPositionLast3Races': [df['DriverAvgRaceFinishPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                    'TeamAvgQualiPositionLast3Races': [df['TeamAvgQualiPositionLast3Races'].mean()] * len(drivers_for_prediction_list),
                    'TeamAvgRaceFinishPositionLast3Races': [df['TeamAvgRaceFinishPositionLast3Races'].mean()] * len(drivers_for_prediction_list)
                })
            else:
                # Recalculate lagged features on this subset of historical data
                window_size = 3
                historical_data_for_prediction.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)

                historical_data_for_prediction['DriverAvgQualiPositionLast3Races'] = historical_data_for_prediction.groupby('Driver')['QualiPosition'].transform(
                    lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
                )
                historical_data_for_prediction['DriverAvgRaceFinishPositionLast3Races'] = historical_data_for_prediction.groupby('Driver')['RaceFinishPosition'].transform(
                    lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
                )
                historical_data_for_prediction['TeamAvgQualiPositionLast3Races'] = historical_data_for_prediction.groupby('Team')['QualiPosition'].transform(
                    lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
                )
                historical_data_for_prediction['TeamAvgRaceFinishPositionLast3Races'] = historical_data_for_prediction.groupby('Team')['RaceFinishPosition'].transform(
                    lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
                )

                # Get the *latest* calculated lagged features for each driver/team from this historical subset
                latest_driver_performance = historical_data_for_prediction.groupby('Driver').last().reset_index()
                latest_team_performance = historical_data_for_prediction.groupby('Team').last().reset_index()

                # Fill any remaining NaNs in these lagged features with the overall mean from the training data
                for col in ['DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
                            'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races']:
                    if latest_driver_performance[col].isnull().any():
                        latest_driver_performance[col].fillna(df[col].mean(), inplace=True)
                    if latest_team_performance[col].isnull().any():
                        latest_team_performance[col].fillna(df[col].mean(), inplace=True)


            # Prepare the input data for prediction
            prediction_data_rows = []
            for i, driver_code in enumerate(drivers_for_prediction_list):
                # Get team from the map or a default if driver not found
                # This logic needs to be more robust for new drivers/teams not in map
                team_name = next((t for d, t in driver_team_map.items() if d == driver_code), "Unknown") if 'driver_team_map' in locals() else "Unknown"
                if team_name == "Unknown" and quali_session_next_race and quali_session_next_race.drivers:
                    # Try to get team from the live session if fallback was not used
                    try:
                        team_name = quali_session_next_race.get_driver(driver_code)['TeamName']
                    except:
                        pass # Keep "Unknown" if not found

                # Get driver's latest historical performance
                driver_hist = latest_driver_performance[latest_driver_performance['Driver'] == driver_code]
                team_hist = latest_team_performance[latest_team_performance['Team'] == team_name]

                # Use mean of the entire training set for AvgLapTime, LapsCompleted, AirTemp, TrackTemp
                # as these are not lagged features and can be approximated by overall averages
                avg_lap_time = df['AvgLapTime'].mean()
                laps_completed = df['LapsCompleted'].mean()
                air_temp = df['AirTemp'].mean()
                track_temp = df['TrackTemp'].mean()
                rainfall = 0 # Assuming no rain for prediction unless a weather forecast API is integrated

                # Get lagged features, using a fallback to overall mean if driver/team not in latest_performance
                driver_avg_quali = driver_hist['DriverAvgQualiPositionLast3Races'].iloc[0] if not driver_hist.empty else df['DriverAvgQualiPositionLast3Races'].mean()
                driver_avg_race = driver_hist['DriverAvgRaceFinishPositionLast3Races'].iloc[0] if not driver_hist.empty else df['DriverAvgRaceFinishPositionLast3Races'].mean()
                team_avg_quali = team_hist['TeamAvgQualiPositionLast3Races'].iloc[0] if not team_hist.empty else df['TeamAvgQualiPositionLast3Races'].mean()
                team_avg_race = team_hist['TeamAvgRaceFinishPositionLast3Races'].iloc[0] if not team_hist.empty else df['TeamAvgRaceFinishPositionLast3Races'].mean()

                prediction_data_rows.append({
                    'Driver': driver_code,
                    'Team': team_name,
                    'QualiPosition': i + 1, # Hypothetical qualifying position 1-20
                    'AvgLapTime': avg_lap_time,
                    'LapsCompleted': laps_completed,
                    'AirTemp': air_temp,
                    'TrackTemp': track_temp,
                    'Rainfall': rainfall,
                    'DriverAvgQualiPositionLast3Races': driver_avg_quali,
                    'DriverAvgRaceFinishPositionLast3Races': driver_avg_race,
                    'TeamAvgQualiPositionLast3Races': team_avg_quali,
                    'TeamAvgRaceFinishPositionLast3Races': team_avg_race
                })

            prediction_input_df = pd.DataFrame(prediction_data_rows)

            # Make prediction using the best model pipeline
            next_race_predictions_raw = best_model_pipeline.predict(prediction_input_df[numerical_features + categorical_features])

            predicted_lineup_df = prediction_input_df.copy()
            predicted_lineup_df['PredictedFinishPosition'] = next_race_predictions_raw
            predicted_lineup_df = predicted_lineup_df.sort_values(by='PredictedFinishPosition').reset_index(drop=True)
            predicted_lineup_df['PredictedRank'] = predicted_lineup_df.index + 1

            print(predicted_lineup_df[['PredictedRank', 'Driver', 'PredictedFinishPosition']].head(20))

    except Exception as e:
        print(f"Error during dynamic next race prediction: {e}")
        print("Please ensure you have collected enough data and FastF1 can access upcoming schedules.")


    # You can also save the trained model for later use
    # import joblib
    # joblib.dump(best_model_pipeline, 'f1_winner_predictor_tuned_pipeline.pkl')
    # print("\nModel saved as f1_winner_predictor_tuned_pipeline.pkl")

if __name__ == "__main__":
    DATA_FILE = "f1_historical_data_with_features.csv"
    train_and_evaluate_model(DATA_FILE)