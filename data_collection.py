import fastf1
import pandas as pd
import os
from datetime import datetime
import pytz # Import pytz for timezone handling
import numpy as np # Import numpy for numerical operations like np.nan

# --- Configuration ---
START_YEAR = 2023
HISTORICAL_DATA_FILE = "f1_historical_data_with_features.csv"
MAIN_CACHE_DIR = 'cache_main'
WINDOW_SIZE = 3 # For lagged features (average over last N races)

# --- Static Track Characteristics (Manually Curated - Add more as needed!) ---
# These are approximate values. You can refine them or add more tracks.
TRACK_CHARACTERISTICS = {
    'Bahrain Grand Prix': {'LengthKm': 5.412, 'NumCorners': 15},
    'Saudi Arabian Grand Prix': {'LengthKm': 6.174, 'NumCorners': 27},
    'Australian Grand Prix': {'LengthKm': 5.278, 'NumCorners': 14},
    'Japanese Grand Prix': {'LengthKm': 5.807, 'NumCorners': 18},
    'Chinese Grand Prix': {'LengthKm': 5.451, 'NumCorners': 16},
    'Miami Grand Prix': {'LengthKm': 5.412, 'NumCorners': 19},
    'Emilia Romagna Grand Prix': {'LengthKm': 4.909, 'NumCorners': 19},
    'Monaco Grand Prix': {'LengthKm': 3.337, 'NumCorners': 19},
    'Spanish Grand Prix': {'LengthKm': 4.657, 'NumCorners': 14},
    'Canadian Grand Prix': {'LengthKm': 4.361, 'NumCorners': 14},
    'Austrian Grand Prix': {'LengthKm': 4.318, 'NumCorners': 10},
    'British Grand Prix': {'LengthKm': 5.891, 'NumCorners': 18},
    'Hungarian Grand Prix': {'LengthKm': 4.381, 'NumCorners': 14},
    'Belgian Grand Prix': {'LengthKm': 7.004, 'NumCorners': 19},
    'Dutch Grand Prix': {'LengthKm': 4.259, 'NumCorners': 14},
    'Italian Grand Prix': {'LengthKm': 5.793, 'NumCorners': 11},
    'Azerbaijan Grand Prix': {'LengthKm': 6.003, 'NumCorners': 20},
    'Singapore Grand Prix': {'LengthKm': 4.940, 'NumCorners': 19},
    'United States Grand Prix': {'LengthKm': 5.513, 'NumCorners': 20},
    'Mexico City Grand Prix': {'LengthKm': 4.304, 'NumCorners': 17},
    'Brazilian Grand Prix': {'LengthKm': 4.309, 'NumCorners': 15},
    'Las Vegas Grand Prix': {'LengthKm': 6.201, 'NumCorners': 17},
    'Abu Dhabi Grand Prix': {'LengthKm': 5.281, 'NumCorners': 16},
    'Qatar Grand Prix': {'LengthKm': 5.419, 'NumCorners': 16},
    # Add more tracks as needed
}


# --- Helper Functions ---

def setup_cache(cache_dir):
    """Creates cache directory if it doesn't exist and enables FastF1 cache."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")
    fastf1.Cache.enable_cache(cache_dir)

def load_session_data(year, round_num, session_type):
    """Loads data for a specific F1 session using round number."""
    print(f"Loading {session_type} session for {year} Round {round_num}...")
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(telemetry=False, laps=True, weather=True)
        print(f"Successfully loaded {session_type} session for {year} Round {round_num}.")
        return session
    except Exception as e:
        print(f"Error loading session {year} Round {round_num} {session_type}: {e}")
        return None

def get_driver_features(session):
    """Extracts basic features for each driver in a session, including full name."""
    if session is None:
        return pd.DataFrame()

    driver_data = []
    for driver_num in session.drivers:
        try:
            driver_info = session.get_driver(driver_num)
            driver_code = driver_info['Abbreviation']
            first_name = driver_info['FirstName']
            last_name = driver_info['LastName']
            full_name = f"{first_name} {last_name}"
            team_name = driver_info['TeamName']

            laps = session.laps.pick_drivers(driver_code)

            if not laps.empty:
                valid_laps = laps.loc[laps['IsAccurate']]
                avg_lap_time = valid_laps['LapTime'].dt.total_seconds().mean() if not valid_laps.empty else None
                laps_completed = len(laps)

                weather_data = session.weather_data
                air_temp = weather_data['AirTemp'].iloc[0] if not weather_data.empty else None
                track_temp = weather_data['TrackTemp'].iloc[0] if not weather_data.empty else None
                rain_status = weather_data['Rainfall'].iloc[0] if not weather_data.empty else 0

                driver_data.append({
                    'Driver': driver_code,
                    'FullName': full_name, # Added FullName
                    'Team': team_name,
                    'AvgLapTime': avg_lap_time,
                    'LapsCompleted': laps_completed,
                    'AirTemp': air_temp,
                    'TrackTemp': track_temp,
                    'Rainfall': rain_status
                })
        except Exception as e:
            print(f"Error processing driver {driver_num} in session {session.event['EventName']}: {e}")
            continue
    return pd.DataFrame(driver_data)

# --- Main Data Collection Logic ---

if __name__ == "__main__":
    current_year = datetime.now().year
    years_to_collect = list(range(START_YEAR, current_year + 1))

    all_races_data = []

    setup_cache(MAIN_CACHE_DIR)

    for year in years_to_collect:
        print(f"\n--- Collecting data for {year} season ---")
        try:
            schedule = fastf1.get_event_schedule(year)

            if schedule['EventDate'].dt.tz is None:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc)
            else:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_convert(pytz.utc)

            if year == current_year:
                now_utc = pd.Timestamp.now(tz='UTC')
                schedule = schedule[schedule['EventDate'] < now_utc]

            for index, event in schedule.iterrows():
                event_name = event['EventName']
                round_num = event['RoundNumber']
                event_date = event['EventDate']

                if round_num == 0:
                    print(f"Skipping Pre-Season Testing for {year}.")
                    continue

                print(f"\nProcessing {year} {event_name} (Round {round_num})...")

                quali_session = load_session_data(year, round_num, 'Q')
                if quali_session is None:
                    print(f"Skipping {event_name} due to missing Qualifying data.")
                    continue

                quali_features_df = get_driver_features(quali_session)

                if not quali_session.results.empty:
                    quali_results = quali_session.results[['Abbreviation', 'Position']].copy()
                    quali_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'}, inplace=True)
                    quali_results['QualiPosition'] = pd.to_numeric(quali_results['QualiPosition'], errors='coerce')
                    quali_features_df = pd.merge(quali_features_df, quali_results, on='Driver', how='left')
                else:
                    print(f"No qualifying results found for {event_name}. QualiPosition will be NaN.")
                    quali_features_df['QualiPosition'] = float('np.nan') # Corrected to np.nan

                race_session = load_session_data(year, round_num, 'R')
                if race_session is None:
                    print(f"Skipping {event_name} due to missing Race data.")
                    continue

                if not race_session.results.empty:
                    race_results = race_session.results[['Abbreviation', 'Position']].copy()
                    race_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'RaceFinishPosition'}, inplace=True)
                    race_results['RaceFinishPosition'] = pd.to_numeric(race_results['RaceFinishPosition'], errors='coerce')
                else:
                    print(f"No race results found for {event_name}. RaceFinishPosition will be NaN.")
                    continue

                merged_df = pd.merge(quali_features_df, race_results, on='Driver', how='inner')

                merged_df['Year'] = year
                merged_df['GrandPrix'] = event_name
                merged_df['EventDate'] = event_date
                merged_df['RoundNumber'] = round_num

                # --- Add Static Track Characteristics ---
                track_info = TRACK_CHARACTERISTICS.get(event_name, {'LengthKm': np.nan, 'NumCorners': np.nan})
                merged_df['TrackLengthKm'] = track_info['LengthKm']
                merged_df['NumCorners'] = track_info['NumCorners']

                all_races_data.append(merged_df)

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue

    if all_races_data:
        final_df = pd.concat(all_races_data, ignore_index=True)

        print("\n--- Engineering historical performance features ---")
        # Sort data chronologically for correct rolling calculations
        final_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)

        # Calculate rolling averages for drivers
        final_df['DriverAvgQualiPositionLast3Races'] = final_df.groupby('Driver')['QualiPosition'].transform(
            lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
        )
        final_df['DriverAvgRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(
            lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
        )

        # Calculate rolling averages for teams
        final_df['TeamAvgQualiPositionLast3Races'] = final_df.groupby('Team')['QualiPosition'].transform(
            lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
        )
        final_df['TeamAvgRaceFinishPositionLast3Races'] = final_df.groupby('Team')['RaceFinishPosition'].transform(
            lambda x: x.shift(1).rolling(window=WINDOW_SIZE, min_periods=1).mean()
        )

        # Calculate time-weighted track-specific average performance (crucial for accuracy!)
        final_df['TrackAvgQualiPosition'] = final_df.groupby('GrandPrix')['QualiPosition'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        final_df['TrackAvgRaceFinishPosition'] = final_df.groupby('GrandPrix')['RaceFinishPosition'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )


        # Fill NaNs for all relevant columns
        for col in ['DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
                    'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races',
                    'TrackAvgQualiPosition', 'TrackAvgRaceFinishPosition',
                    'TrackLengthKm', 'NumCorners']: # Added new track features here
            if final_df[col].isnull().any():
                fill_value = final_df[col].mean()
                final_df[col] = final_df[col].fillna(fill_value)
                print(f"Filled NaN in {col} with mean: {fill_value:.2f}")


        print("\n--- Consolidated Historical Data with New Features (Sample) ---")
        print(final_df.head())
        print(f"\nTotal rows collected: {len(final_df)}")

        final_df.to_csv(HISTORICAL_DATA_FILE, index=False)
        print(f"\nConsolidated data saved to {HISTORICAL_DATA_FILE}")
    else:
        print("\nNo data collected for the specified years. Please check for errors during data loading.")
