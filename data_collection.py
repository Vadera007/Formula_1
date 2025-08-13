import pandas as pd
import sqlite3
import os
import fastf1
from datetime import datetime
import pytz
import numpy as np
from fastf1.core import DataNotLoadedError

# --- Configuration ---
START_YEAR = 2018
DB_FILE = "f1_predictor.db"
TABLE_NAME = "historical_data"
MAIN_CACHE_DIR = 'cache_main'
WINDOW_SIZE = 3

# --- Static Track Characteristics & Types ---
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

def get_existing_races(db_file, table_name):
    if not os.path.exists(db_file): return set(), pd.DataFrame()
    try:
        conn = sqlite3.connect(db_file)
        query = f"SELECT * FROM {table_name}"
        existing_df = pd.read_sql_query(query, conn)
        conn.close()
        existing_races_set = {tuple(x) for x in existing_df[['Year', 'RoundNumber']].to_numpy()}
        print(f"Found {len(existing_races_set)} existing races in the database.")
        return existing_races_set, existing_df
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        print(f"Database is empty or corrupt. Will build from scratch. Error: {e}")
        return set(), pd.DataFrame()

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
                if not session.laps.empty: all_practice_laps_list.append(session.laps)
            except DataNotLoadedError:
                print(f"Practice session {fp} for {year} Round {round_num} has no lap data. Skipping.")
                continue
    if not all_practice_laps_list: return pd.DataFrame()
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

def process_new_races(existing_races_set):
    newly_processed_data = []
    current_year = datetime.now().year
    years_to_collect = list(range(START_YEAR, current_year + 1))
    
    for year in years_to_collect:
        print(f"\n--- Checking {year} season for new data ---")
        try:
            schedule = fastf1.get_event_schedule(year)
        except ValueError:
            print(f"Could not load schedule for {year}. Skipping.")
            continue
        
        schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc) if schedule['EventDate'].dt.tz is None else schedule['EventDate'].dt.tz_convert(pytz.utc)
        schedule_to_process = schedule[schedule['EventDate'] < pd.Timestamp.now(tz='UTC')]

        for _, event in schedule_to_process.iterrows():
            event_name, round_num = event['EventName'], event['RoundNumber']
            race_identifier = (year, round_num)

            if "testing" in event_name.lower() or round_num == 0: continue
            if race_identifier in existing_races_set:
                # print(f"SKIPPING: {year} {event_name}, data already exists.") # Uncomment for verbose skipping
                continue
            
            print(f"PROCESSING: {year} {event_name}...")
            # --- Start of single race processing ---
            quali_session = load_session_data(year, round_num, 'Q')
            if quali_session is None: continue
            
            quali_features_df = get_driver_features(quali_session)
            if quali_features_df.empty: continue

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
            if teammate_battles: race_results = pd.merge(race_results, pd.DataFrame(teammate_battles), on='Driver', how='left')

            merged_df = pd.merge(quali_features_df, race_results, on=['Driver', 'Team'], how='inner')
            if not practice_features_df.empty: merged_df = pd.merge(merged_df, practice_features_df, on='Driver', how='left')
            
            merged_df['NumPitStops'], merged_df['AvgStintLength'], merged_df['MedianLapTime'], merged_df['StdDevLapTime'], merged_df['AvgSector2Pace'], merged_df['StartingTyreCompound'] = [np.nan, np.nan, np.nan, np.nan, np.nan, 'UNKNOWN']

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
            merged_df['TrackLengthKm'], merged_df['NumCorners'] = track_info.get('LengthKm', np.nan), track_info.get('NumCorners', np.nan)
            newly_processed_data.append(merged_df)
            # --- End of single race processing ---
    return newly_processed_data

def engineer_features_and_save(existing_df, new_data_list):
    print("\n--- Engineering features for the full dataset ---")
    if not new_data_list and existing_df.empty:
        print("No data available to process.")
        return
    
    full_df = pd.concat([existing_df] + new_data_list, ignore_index=True)
    full_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
    
    # --- Engineer All Rolling/Expanding Features ---
    full_df['DriverAvgQualiPositionLast3Races'] = full_df.groupby('Driver')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
    full_df['DriverAvgRaceFinishPositionLast3Races'] = full_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
    full_df['DriverStdDevRaceFinishPositionLast3Races'] = full_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).std()).fillna(0)
    full_df['TeamAvgQualiPositionLast3Races'] = full_df.groupby('Team')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
    full_df['TeamAvgRaceFinishPositionLast3Races'] = full_df.groupby('Team')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
    full_df['TrackAvgQualiPosition'] = full_df.groupby('GrandPrix')['QualiPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    full_df['TrackAvgRaceFinishPosition'] = full_df.groupby('GrandPrix')['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    full_df['TeammateWinRate'] = full_df.groupby('Driver')['BeatTeammate'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    full_df['DriverDNFRate'] = full_df.groupby('Driver')['IsDNF'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    full_df['TrackTypePerformance'] = full_df.groupby(['Driver', 'TrackType'])['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    cols_to_fill = [
        'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races',
        'TrackAvgQualiPosition', 'TrackAvgRaceFinishPosition', 'NumPitStops', 'MedianLapTime',
        'StdDevLapTime', 'AvgSector2Pace', 'PracticeFastestLap', 'AvgLongRunPace',
        'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance', 'BeatTeammate',
        'TrackLengthKm', 'NumCorners', 'AvgStintLength'
    ]
    for col in cols_to_fill:
        if col in full_df.columns and full_df[col].isnull().any():
            full_df[col] = full_df[col].fillna(full_df[col].mean())
    
    # --- FIX: Convert Timestamp to string before saving to SQLite ---
    if 'EventDate' in full_df.columns:
        full_df['EventDate'] = full_df['EventDate'].astype(str)

    conn = sqlite3.connect(DB_FILE)
    full_df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print(f"\n✅ Consolidated data with all new features saved to SQLite database: {DB_FILE}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    setup_cache(MAIN_CACHE_DIR)
    existing_races_set, existing_df = get_existing_races(DB_FILE, TABLE_NAME)
    new_data_list = process_new_races(existing_races_set)
    
    if new_data_list:
        engineer_features_and_save(existing_df, new_data_list)
    else:
        print("\n✅ No new races to add. Database is already up-to-date.")




# import fastf1
# import pandas as pd
# import os
# from datetime import datetime
# import pytz 
# import numpy as np 
# from fastf1.core import DataNotLoadedError
# import sqlite3

# # --- Configuration ---
# START_YEAR = 2018
# DB_FILE = "f1_predictor.db" 
# MAIN_CACHE_DIR = 'cache_main'
# WINDOW_SIZE = 3 

# # --- Static Track Characteristics ---
# TRACK_CHARACTERISTICS = {
#     'Bahrain Grand Prix': {'LengthKm': 5.412, 'NumCorners': 15}, 'Saudi Arabian Grand Prix': {'LengthKm': 6.174, 'NumCorners': 27},
#     'Australian Grand Prix': {'LengthKm': 5.278, 'NumCorners': 14}, 'Japanese Grand Prix': {'LengthKm': 5.807, 'NumCorners': 18},
#     'Chinese Grand Prix': {'LengthKm': 5.451, 'NumCorners': 16}, 'Miami Grand Prix': {'LengthKm': 5.412, 'NumCorners': 19},
#     'Emilia Romagna Grand Prix': {'LengthKm': 4.909, 'NumCorners': 19}, 'Monaco Grand Prix': {'LengthKm': 3.337, 'NumCorners': 19},
#     'Spanish Grand Prix': {'LengthKm': 4.657, 'NumCorners': 14}, 'Canadian Grand Prix': {'LengthKm': 4.361, 'NumCorners': 14},
#     'Austrian Grand Prix': {'LengthKm': 4.318, 'NumCorners': 10}, 'British Grand Prix': {'LengthKm': 5.891, 'NumCorners': 18},
#     'Hungarian Grand Prix': {'LengthKm': 4.381, 'NumCorners': 14}, 'Belgian Grand Prix': {'LengthKm': 7.004, 'NumCorners': 19},
#     'Dutch Grand Prix': {'LengthKm': 4.259, 'NumCorners': 14}, 'Italian Grand Prix': {'LengthKm': 5.793, 'NumCorners': 11},
#     'Azerbaijan Grand Prix': {'LengthKm': 6.003, 'NumCorners': 20}, 'Singapore Grand Prix': {'LengthKm': 4.940, 'NumCorners': 19},
#     'United States Grand Prix': {'LengthKm': 5.513, 'NumCorners': 20}, 'Mexico City Grand Prix': {'LengthKm': 4.304, 'NumCorners': 17},
#     'Brazilian Grand Prix': {'LengthKm': 4.309, 'NumCorners': 15}, 'Las Vegas Grand Prix': {'LengthKm': 6.201, 'NumCorners': 17},
#     'Abu Dhabi Grand Prix': {'LengthKm': 5.281, 'NumCorners': 16}, 'Qatar Grand Prix': {'LengthKm': 5.419, 'NumCorners': 16},
#     'French Grand Prix': {'LengthKm': 5.842, 'NumCorners': 15}, 'Styrian Grand Prix': {'LengthKm': 4.318, 'NumCorners': 10},
#     'Tuscan Grand Prix': {'LengthKm': 5.245, 'NumCorners': 15}, 'Eifel Grand Prix': {'LengthKm': 5.148, 'NumCorners': 15},
#     'Portuguese Grand Prix': {'LengthKm': 4.653, 'NumCorners': 15}, 'Turkish Grand Prix': {'LengthKm': 5.338, 'NumCorners': 14},
#     'German Grand Prix': {'LengthKm': 4.574, 'NumCorners': 17}, 'Russian Grand Prix': {'LengthKm': 5.848, 'NumCorners': 18},
# }

# # --- Track Type Classification ---
# TRACK_TYPES = {
#     'Bahrain Grand Prix': 'Balanced', 'Saudi Arabian Grand Prix': 'Street Circuit', 'Australian Grand Prix': 'Street Circuit',
#     'Japanese Grand Prix': 'High Downforce', 'Chinese Grand Prix': 'Balanced', 'Miami Grand Prix': 'Street Circuit',
#     'Emilia Romagna Grand Prix': 'High Downforce', 'Monaco Grand Prix': 'Street Circuit', 'Spanish Grand Prix': 'High Downforce',
#     'Canadian Grand Prix': 'Low Downforce', 'Austrian Grand Prix': 'Low Downforce', 'British Grand Prix': 'Low Downforce',
#     'Hungarian Grand Prix': 'High Downforce', 'Belgian Grand Prix': 'Low Downforce', 'Dutch Grand Prix': 'High Downforce',
#     'Italian Grand Prix': 'Low Downforce', 'Azerbaijan Grand Prix': 'Street Circuit', 'Singapore Grand Prix': 'Street Circuit',
#     'United States Grand Prix': 'Balanced', 'Mexico City Grand Prix': 'High Altitude', 'Brazilian Grand Prix': 'High Altitude',
#     'Las Vegas Grand Prix': 'Street Circuit', 'Abu Dhabi Grand Prix': 'Balanced', 'Qatar Grand Prix': 'Balanced',
#     'French Grand Prix': 'Balanced', 'Styrian Grand Prix': 'Low Downforce', 'Tuscan Grand Prix': 'High Downforce',
#     'Eifel Grand Prix': 'Balanced', 'Portuguese Grand Prix': 'Balanced', 'Turkish Grand Prix': 'High Downforce',
#     'German Grand Prix': 'Balanced', 'Russian Grand Prix': 'Street Circuit'
# }

# # --- Helper Functions ---
# def setup_cache(cache_dir):
#     if not os.path.exists(cache_dir): os.makedirs(cache_dir)
#     fastf1.Cache.enable_cache(cache_dir)

# def load_session_data(year, round_num, session_type):
#     try:
#         session = fastf1.get_session(year, round_num, session_type)
#         session.load(telemetry=False, laps=True, weather=True)
#         return session
#     except Exception as e:
#         print(f"Could not load {session_type} for {year} Round {round_num}: {e}")
#         return None

# def get_driver_features(session):
#     if session is None: return pd.DataFrame()
#     driver_data = []
#     for driver_num in session.drivers:
#         try:
#             driver_info = session.get_driver(driver_num)
#             laps = session.laps.pick_drivers([driver_info['Abbreviation']])
#             if not laps.empty:
#                 valid_laps = laps.loc[laps['IsAccurate']]
#                 weather = session.weather_data
#                 driver_data.append({
#                     'Driver': driver_info['Abbreviation'], 'FullName': f"{driver_info['FirstName']} {driver_info['LastName']}", 'Team': driver_info['TeamName'],
#                     'AvgLapTime': valid_laps['LapTime'].dt.total_seconds().median() if not valid_laps.empty else np.nan,
#                     'LapsCompleted': len(laps), 'AirTemp': weather['AirTemp'].iloc[0] if not weather.empty else np.nan,
#                     'TrackTemp': weather['TrackTemp'].iloc[0] if not weather.empty else np.nan,
#                     'Rainfall': weather['Rainfall'].iloc[0] if not weather.empty else 0
#                 })
#         except Exception as e:
#             print(f"Error processing driver {driver_num}: {e}")
#     return pd.DataFrame(driver_data)

# def get_practice_features(year, round_num, race_drivers_list):
#     all_practice_laps_list = []
#     for fp in ['FP1', 'FP2', 'FP3']:
#         session = load_session_data(year, round_num, fp)
#         if session:
#             try:
#                 laps = session.laps
#                 if not laps.empty:
#                     all_practice_laps_list.append(laps)
#             except DataNotLoadedError:
#                 print(f"Practice session {fp} for {year} Round {round_num} has no lap data. Skipping.")
#                 continue
    
#     if not all_practice_laps_list:
#         return pd.DataFrame()

#     practice_laps = pd.concat(all_practice_laps_list)
#     if practice_laps.empty: return pd.DataFrame()
    
#     practice_features = []
#     for driver in race_drivers_list:
#         driver_laps = practice_laps.pick_drivers([driver])
#         if driver_laps.empty: continue
#         stints = driver_laps.groupby('Stint')
#         long_run_paces = [s['LapTime'].median().total_seconds() for _, s in stints if len(s) >= 5]
#         practice_features.append({
#             'Driver': driver,
#             'PracticeFastestLap': driver_laps['LapTime'].min().total_seconds() if not driver_laps.empty else np.nan,
#             'AvgLongRunPace': min(long_run_paces) if long_run_paces else np.nan
#         })
#     return pd.DataFrame(practice_features)

# # --- Main Data Collection Logic ---
# if __name__ == "__main__":
#     current_year = datetime.now().year
#     years_to_collect = list(range(START_YEAR, current_year + 1)) 
#     all_races_data = []
#     setup_cache(MAIN_CACHE_DIR)
#     for year in years_to_collect:
#         print(f"\n--- Collecting data for {year} season ---")
        
#         # --- Handle years with no schedule data ---
#         try:
#             schedule = fastf1.get_event_schedule(year)
#         except ValueError:
#             print(f"Could not load schedule for {year}. It may not be available yet. Skipping.")
#             continue # Skip to the next year
            
#         schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc) if schedule['EventDate'].dt.tz is None else schedule['EventDate'].dt.tz_convert(pytz.utc)
        
#         # Filter to only include races that have already happened
#         schedule = schedule[schedule['EventDate'] < pd.Timestamp.now(tz='UTC')]

#         for index, event in schedule.iterrows():
#             event_name, round_num = event['EventName'], event['RoundNumber']
#             if "testing" in event_name.lower() or round_num == 0: continue
#             print(f"\nProcessing {year} {event_name}...")
#             quali_session = load_session_data(year, round_num, 'Q')
#             if quali_session is None: continue
            
#             quali_features_df = get_driver_features(quali_session)
#             if quali_features_df.empty:
#                 print(f"No driver data extracted for qualifying in {year} {event_name}. Skipping race.")
#                 continue

#             race_drivers_list = quali_features_df['Driver'].unique().tolist()
#             practice_features_df = get_practice_features(year, round_num, race_drivers_list)
            
#             if not quali_session.results.empty:
#                 quali_results = quali_session.results[['Abbreviation', 'Position']].rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'})
#                 quali_features_df = pd.merge(quali_features_df, quali_results, on='Driver', how='left')
            
#             race_session = load_session_data(year, round_num, 'R')
#             if race_session is None or race_session.results.empty: continue
            
#             race_results = race_session.results[['Abbreviation', 'Position', 'TeamName', 'Status']].rename(columns={'Abbreviation': 'Driver', 'Position': 'RaceFinishPosition', 'TeamName': 'Team'})
#             race_results['IsDNF'] = race_results['Status'].apply(lambda x: 0 if 'Lap' in x or 'Finished' in x else 1)
            
#             teammate_battles = []
#             for team in race_results['Team'].unique():
#                 team_drivers = race_results[race_results['Team'] == team]
#                 if len(team_drivers) == 2:
#                     d1, d2 = team_drivers.iloc[0], team_drivers.iloc[1]
#                     winner, loser = (d1, d2) if d1['RaceFinishPosition'] < d2['RaceFinishPosition'] else (d2, d1)
#                     teammate_battles.append({'Driver': winner['Driver'], 'BeatTeammate': 1})
#                     teammate_battles.append({'Driver': loser['Driver'], 'BeatTeammate': 0})
#             if teammate_battles:
#                 race_results = pd.merge(race_results, pd.DataFrame(teammate_battles), on='Driver', how='left')

#             merged_df = pd.merge(quali_features_df, race_results, on=['Driver', 'Team'], how='inner')
#             if not practice_features_df.empty:
#                 merged_df = pd.merge(merged_df, practice_features_df, on='Driver', how='left')
            
#             for col in ['NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace']:
#                 merged_df[col] = np.nan
#             merged_df['StartingTyreCompound'] = 'UNKNOWN'

#             try:
#                 if not race_session.laps.empty:
#                     for driver_code in merged_df['Driver']:
#                         laps = race_session.laps.pick_drivers([driver_code])
#                         if not laps.empty:
#                             accurate_laps = laps.loc[laps['IsAccurate']]
#                             num_pits = len(laps.loc[laps['PitInTime'].notna()])
#                             laps_completed = merged_df.loc[merged_df['Driver'] == driver_code, 'LapsCompleted'].iloc[0]
#                             merged_df.loc[merged_df['Driver'] == driver_code, 'NumPitStops'] = num_pits
#                             merged_df.loc[merged_df['Driver'] == driver_code, 'AvgStintLength'] = laps_completed / (num_pits + 1) if laps_completed > 0 else 0
#                             merged_df.loc[merged_df['Driver'] == driver_code, 'StartingTyreCompound'] = laps.pick_track_status('1')['Compound'].iloc[0] if not laps.pick_track_status('1').empty else 'UNKNOWN'
#                             if not accurate_laps.empty:
#                                 merged_df.loc[merged_df['Driver'] == driver_code, 'MedianLapTime'] = accurate_laps['LapTime'].dt.total_seconds().median()
#                                 merged_df.loc[merged_df['Driver'] == driver_code, 'StdDevLapTime'] = accurate_laps['LapTime'].dt.total_seconds().std()
#                                 merged_df.loc[merged_df['Driver'] == driver_code, 'AvgSector2Pace'] = accurate_laps['Sector2Time'].dt.total_seconds().mean()
#             except DataNotLoadedError:
#                  print(f"Race session for {year} {event_name} has no lap data. Skipping strategy features.")

#             merged_df['Year'], merged_df['GrandPrix'], merged_df['EventDate'], merged_df['RoundNumber'] = year, event_name, event['EventDate'], round_num
#             merged_df['TrackType'] = TRACK_TYPES.get(event_name, 'Unknown')
#             track_info = TRACK_CHARACTERISTICS.get(event_name, {})
#             merged_df['TrackLengthKm'] = track_info.get('LengthKm', np.nan)
#             merged_df['NumCorners'] = track_info.get('NumCorners', np.nan)
#             all_races_data.append(merged_df)

#     if all_races_data:
#         final_df = pd.concat(all_races_data, ignore_index=True)
#         final_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)
        
#         # --- Engineer All Rolling/Expanding Features ---
#         final_df['DriverAvgQualiPositionLast3Races'] = final_df.groupby('Driver')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
#         final_df['DriverAvgRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
#         final_df['DriverStdDevRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).std()).fillna(0)
#         final_df['TeamAvgQualiPositionLast3Races'] = final_df.groupby('Team')['QualiPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
#         final_df['TeamAvgRaceFinishPositionLast3Races'] = final_df.groupby('Team')['RaceFinishPosition'].transform(lambda x: x.shift(1).rolling(WINDOW_SIZE, min_periods=1).mean())
#         final_df['TrackAvgQualiPosition'] = final_df.groupby('GrandPrix')['QualiPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
#         final_df['TrackAvgRaceFinishPosition'] = final_df.groupby('GrandPrix')['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
#         final_df['TeammateWinRate'] = final_df.groupby('Driver')['BeatTeammate'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
#         final_df['DriverDNFRate'] = final_df.groupby('Driver')['IsDNF'].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
#         final_df['TrackTypePerformance'] = final_df.groupby(['Driver', 'TrackType'])['RaceFinishPosition'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

#         cols_to_fill = [
#             'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races',
#             'TrackAvgQualiPosition', 'TrackAvgRaceFinishPosition', 'NumPitStops', 'MedianLapTime',
#             'StdDevLapTime', 'AvgSector2Pace', 'PracticeFastestLap', 'AvgLongRunPace',
#             'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance', 'BeatTeammate',
#             'TrackLengthKm', 'NumCorners', 'AvgStintLength'
#         ]
#         for col in cols_to_fill:
#             if col in final_df.columns and final_df[col].isnull().any():
#                 final_df[col] = final_df[col].fillna(final_df[col].mean())
        
#         conn = sqlite3.connect(DB_FILE)
#         final_df.to_sql('historical_data', conn, if_exists='replace', index=False)
#         conn.close()
#         print(f"\nConsolidated data with all new features saved to SQLite database: {DB_FILE}")
