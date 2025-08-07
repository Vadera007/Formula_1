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
        print(f"Could not load {session_type} for {year} Round {round_num}: {e}")
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
            full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"
            team_name = driver_info['TeamName']
            laps = session.laps.pick_drivers([driver_code])

            if not laps.empty:
                valid_laps = laps.loc[laps['IsAccurate']]
                avg_lap_time = valid_laps['LapTime'].dt.total_seconds().median() if not valid_laps.empty else np.nan
                laps_completed = len(laps)
                weather_data = session.weather_data
                air_temp = weather_data['AirTemp'].iloc[0] if not weather_data.empty else None
                track_temp = weather_data['TrackTemp'].iloc[0] if not weather_data.empty else None
                rain_status = weather_data['Rainfall'].iloc[0] if not weather_data.empty else 0

                driver_data.append({
                    'Driver': driver_code, 'FullName': full_name, 'Team': team_name,
                    'AvgLapTime': avg_lap_time, 'LapsCompleted': laps_completed,
                    'AirTemp': air_temp, 'TrackTemp': track_temp, 'Rainfall': rain_status
                })
        except Exception as e:
            print(f"Error processing driver {driver_num} in session {session.event['EventName']}: {e}")
            continue
    return pd.DataFrame(driver_data)

def get_practice_features(year, round_num, race_drivers_list):
    """
    Loads all practice sessions for a race weekend and calculates key performance features.
    - PracticeFastestLap: The best single lap time across all FP sessions.
    - AvgLongRunPace: The median pace from the driver's best long run (stint > 4 laps).
    """
    practice_sessions = []
    for fp in ['FP1', 'FP2', 'FP3']:
        session = load_session_data(year, round_num, fp)
        if session:
            practice_sessions.append(session.laps)
    
    if not practice_sessions:
        print("No practice sessions found. Skipping practice feature extraction.")
        return pd.DataFrame()

    all_practice_laps = pd.concat(practice_sessions)
    practice_features = []

    for driver in race_drivers_list:
        driver_laps = all_practice_laps.pick_drivers([driver])
        if driver_laps.empty:
            continue

        fastest_lap = driver_laps['LapTime'].min().total_seconds() if not driver_laps.empty else np.nan

        long_run_paces = []
        stints = driver_laps.groupby('Stint')
        for _, stint_laps in stints:
            if len(stint_laps) >= 5:
                long_run_pace = stint_laps['LapTime'].median().total_seconds()
                long_run_paces.append(long_run_pace)
        
        avg_long_run_pace = min(long_run_paces) if long_run_paces else np.nan

        practice_features.append({
            'Driver': driver,
            'PracticeFastestLap': fastest_lap,
            'AvgLongRunPace': avg_long_run_pace
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
        try:
            schedule = fastf1.get_event_schedule(year)
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc) if schedule['EventDate'].dt.tz is None else schedule['EventDate'].dt.tz_convert(pytz.utc)
            if year == current_year:
                now_utc = pd.Timestamp.now(tz='UTC')
                schedule = schedule[schedule['EventDate'] < now_utc]

            for index, event in schedule.iterrows():
                event_name, round_num = event['EventName'], event['RoundNumber']
                if "testing" in event_name.lower() or round_num == 0:
                    continue
                print(f"\nProcessing {year} {event_name} (Round {round_num})...")

                quali_session = load_session_data(year, round_num, 'Q')
                if quali_session is None:
                    print(f"Skipping {event_name} due to missing Qualifying data.")
                    continue

                race_drivers_list = [quali_session.get_driver(d)['Abbreviation'] for d in quali_session.drivers]
                practice_features_df = get_practice_features(year, round_num, race_drivers_list)
                quali_features_df = get_driver_features(quali_session)

                if not quali_session.results.empty:
                    quali_results = quali_session.results[['Abbreviation', 'Position']].copy()
                    quali_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'}, inplace=True)
                    quali_features_df = pd.merge(quali_features_df, quali_results, on='Driver', how='left')
                
                race_session = load_session_data(year, round_num, 'R')
                if race_session is None: continue
                if not race_session.results.empty:
                    race_results = race_session.results[['Abbreviation', 'Position']].copy()
                    race_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'RaceFinishPosition'}, inplace=True)
                else: continue

                merged_df = pd.merge(quali_features_df, race_results, on='Driver', how='inner')
                if not practice_features_df.empty:
                    merged_df = pd.merge(merged_df, practice_features_df, on='Driver', how='left')

                print("Calculating Tier 1 Strategy and Pace Features...")
                tier1_features_list = []
                for driver_code in merged_df['Driver']:
                    try:
                        laps = race_session.laps.pick_drivers([driver_code])
                        num_pits = len(laps.loc[laps['PitInTime'].notna()])
                        accurate_laps = laps.loc[laps['IsAccurate']]
                        starting_compound = laps.pick_track_status('1')['Compound'].iloc[0] if not laps.pick_track_status('1').empty else 'UNKNOWN'
                        laps_completed_val = merged_df.loc[merged_df['Driver'] == driver_code, 'LapsCompleted'].iloc[0]
                        avg_stint_length = laps_completed_val / (num_pits + 1) if laps_completed_val > 0 else 0
                        tier1_features_list.append({
                            'Driver': driver_code, 'NumPitStops': num_pits, 'StartingTyreCompound': starting_compound,
                            'AvgStintLength': avg_stint_length,
                            'MedianLapTime': accurate_laps['LapTime'].dt.total_seconds().median() if not accurate_laps.empty else np.nan,
                            'StdDevLapTime': accurate_laps['LapTime'].dt.total_seconds().std() if not accurate_laps.empty else np.nan,
                            'AvgSector2Pace': accurate_laps['Sector2Time'].dt.total_seconds().mean() if not accurate_laps.empty else np.nan,
                        })
                    except Exception as e:
                        print(f"Could not calculate Tier 1 features for {driver_code}: {e}")
                
                if tier1_features_list:
                    tier1_df = pd.DataFrame(tier1_features_list)
                    merged_df = pd.merge(merged_df, tier1_df, on='Driver', how='left')
                
                merged_df['Year'] = year
                merged_df['GrandPrix'] = event_name
                merged_df['EventDate'] = event['EventDate']
                merged_df['RoundNumber'] = round_num
                track_info = TRACK_CHARACTERISTICS.get(event_name, {})
                merged_df['TrackLengthKm'] = track_info.get('LengthKm', np.nan)
                merged_df['NumCorners'] = track_info.get('NumCorners', np.nan)
                all_races_data.append(merged_df)
        except Exception as e:
            print(f"Critical error processing year {year}: {e}")

    if all_races_data:
        final_df = pd.concat(all_races_data, ignore_index=True)
        final_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
        print("\n--- Engineering historical performance features ---")
        
        # Lagged Features
        final_df['DriverAvgQualiPositionLast3Races'] = final_df.groupby('Driver')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['DriverAvgRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['DriverStdDevRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).std()).fillna(0)
        final_df['TeamAvgQualiPositionLast3Races'] = final_df.groupby('Team')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        final_df['TeamAvgRaceFinishPositionLast3Races'] = final_df.groupby('Team')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
        
        # Expanding Track Averages
        final_df['TrackAvgQualiPosition'] = final_df.groupby('GrandPrix')['QualiPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        final_df['TrackAvgRaceFinishPosition'] = final_df.groupby('GrandPrix')['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        
        # NaN Filling
        cols_to_fill = [
            'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
            'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races',
            'TrackAvgQualiPosition', 'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
            'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
            'PracticeFastestLap', 'AvgLongRunPace'
        ]
        for col in cols_to_fill:
            if col in final_df.columns and final_df[col].isnull().any():
                fill_value = 0 if 'StdDev' in col else final_df[col].mean()
                # --- CORRECTED LINE TO AVOID WARNING ---
                final_df[col] = final_df[col].fillna(fill_value)
                print(f"Filled NaN in {col} with value: {fill_value:.2f}")

        print("\n--- Consolidated Historical Data with New Features (Sample) ---")
        print(final_df[['Driver', 'RaceFinishPosition', 'PracticeFastestLap', 'AvgLongRunPace']].head())
        final_df.to_csv(HISTORICAL_DATA_FILE, index=False)
        print(f"\nConsolidated data saved to {HISTORICAL_DATA_FILE}")
    else:
        print("\nNo data collected. Please check for errors.")

