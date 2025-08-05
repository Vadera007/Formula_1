import fastf1
import pandas as pd
import os
from datetime import datetime
import pytz # Import pytz for timezone handling

# Define the cache directory path
CACHE_DIR = 'cache'

# Create the cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    print(f"Created cache directory: {CACHE_DIR}")

# Enable caching for faster data loading on subsequent runs
fastf1.Cache.enable_cache(CACHE_DIR)

def load_session_data(year, round_num, session_type):
    """
    Loads data for a specific F1 session using round number.

    Args:
        year (int): The year of the Grand Prix.
        round_num (int): The round number of the Grand Prix.
        session_type (str): The session type ('FP1', 'FP2', 'FP3', 'Q', 'R').

    Returns:
        fastf1.core.Session: The loaded FastF1 session object.
    """
    print(f"Loading {session_type} session for {year} Round {round_num}...")
    try:
        session = fastf1.get_session(year, round_num, session_type)
        # Load laps and weather data. Telemetry can be very large, so we'll skip it for now.
        session.load(telemetry=False, laps=True, weather=True)
        print(f"Successfully loaded {session_type} session for {year} Round {round_num}.")
        return session
    except Exception as e:
        print(f"Error loading session {year} Round {round_num} {session_type}: {e}")
        return None

def get_driver_features(session):
    """
    Extracts basic features for each driver in a session.
    This function does NOT extract Qualifying Position, as it's best obtained
    directly from session.results for 'Q' sessions.

    Args:
        session (fastf1.core.Session): The loaded FastF1 session object.

    Returns:
        pd.DataFrame: A DataFrame with engineered features for each driver,
                      excluding 'QualiPosition' for now.
    """
    if session is None:
        return pd.DataFrame()

    driver_data = []
    for driver_num in session.drivers:
        try:
            driver_info = session.get_driver(driver_num)
            driver_code = driver_info['Abbreviation']
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
                    'Team': team_name,
                    'AvgLapTime': avg_lap_time,
                    'LapsCompleted': laps_completed,
                    'AirTemp': air_temp,
                    'TrackTemp': track_temp,
                    'Rainfall': rain_status
                })
        except Exception as e:
            print(f"Error processing driver {driver_num} in session {session.event['EventName']}: {e}")
            continue # Skip to the next driver if an error occurs

    return pd.DataFrame(driver_data)

if __name__ == "__main__":
    START_YEAR = 2023
    CURRENT_YEAR = datetime.now().year
    # Include current year if it's not a full season yet, only completed races
    YEARS_TO_COLLECT = list(range(START_YEAR, CURRENT_YEAR + 1))

    all_races_data = []

    for year in YEARS_TO_COLLECT:
        print(f"\n--- Collecting data for {year} season ---")
        try:
            schedule = fastf1.get_event_schedule(year)

            # Convert EventDate to timezone-aware UTC for consistent comparison
            if schedule['EventDate'].dt.tz is None:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_localize(pytz.utc)
            else:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_convert(pytz.utc)

            # Filter for 'Race' events (which include Qualifying sessions)
            if year == CURRENT_YEAR:
                now_utc = pd.Timestamp.now(tz='UTC')
                schedule = schedule[schedule['EventDate'] < now_utc]

            # Iterate through each event in the schedule
            for index, event in schedule.iterrows():
                event_name = event['EventName']
                round_num = event['RoundNumber']
                event_date = event['EventDate']

                # Skip pre-season testing events (Round 0) as they often don't have 'Q' or 'R' sessions
                if round_num == 0:
                    print(f"Skipping Pre-Season Testing for {year}.")
                    continue

                print(f"\nProcessing {year} {event_name} (Round {round_num})...")

                # Load Qualifying data
                quali_session = load_session_data(year, round_num, 'Q')
                if quali_session is None:
                    print(f"Skipping {event_name} due to missing Qualifying data.")
                    continue

                quali_features_df = get_driver_features(quali_session)

                # Get official qualifying positions and merge them
                if not quali_session.results.empty:
                    quali_results = quali_session.results[['Abbreviation', 'Position']].copy()
                    quali_results.rename(columns={'Abbreviation': 'Driver', 'Position': 'QualiPosition'}, inplace=True)
                    quali_results['QualiPosition'] = pd.to_numeric(quali_results['QualiPosition'], errors='coerce')
                    quali_features_df = pd.merge(quali_features_df, quali_results, on='Driver', how='left')
                else:
                    print(f"No qualifying results found for {event_name}. QualiPosition will be NaN.")
                    quali_features_df['QualiPosition'] = float('NaN')

                # Load Race data to get actual finishing positions (our target variable)
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
                    continue # Skip this race if no race results are available

                # Merge features and results
                merged_df = pd.merge(quali_features_df, race_results, on='Driver', how='inner')

                # Add Year, GrandPrix, and RoundNumber for identification and historical feature calculation
                merged_df['Year'] = year
                merged_df['GrandPrix'] = event_name
                merged_df['EventDate'] = event_date
                merged_df['RoundNumber'] = round_num # Add RoundNumber for sorting

                all_races_data.append(merged_df)

        except Exception as e:
            print(f"Error processing year {year}: {e}")
            continue # Continue to the next year if an error occurs

    if all_races_data:
        # Concatenate all individual race DataFrames into one large DataFrame
        final_df = pd.concat(all_races_data, ignore_index=True)

        # --- Advanced Feature Engineering: Lagged Features ---
        print("\n--- Engineering historical performance features ---")
        # Sort data chronologically for correct rolling calculations
        final_df.sort_values(by=['EventDate', 'RoundNumber', 'Driver'], inplace=True)

        # Calculate rolling averages for drivers
        # Shift by 1 to ensure we only use data from *previous* races
        window_size = 3 # Average over the last 3 races

        final_df['DriverAvgQualiPositionLast3Races'] = final_df.groupby('Driver')['QualiPosition'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )
        final_df['DriverAvgRaceFinishPositionLast3Races'] = final_df.groupby('Driver')['RaceFinishPosition'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )

        # Calculate rolling averages for teams
        final_df['TeamAvgQualiPositionLast3Races'] = final_df.groupby('Team')['QualiPosition'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )
        final_df['TeamAvgRaceFinishPositionLast3Races'] = final_df.groupby('Team')['RaceFinishPosition'].transform(
            lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
        )

        # Handle NaN values created by shifting/rolling (e.g., for first few races)
        # For simplicity, we'll fill with the overall mean, but more sophisticated methods exist.
        for col in ['DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
                    'TeamAvgQualiPositionLast3Races', 'TeamAvgRaceFinishPositionLast3Races']:
            if final_df[col].isnull().any():
                fill_value = final_df[col].mean()
                final_df[col].fillna(fill_value, inplace=True)
                print(f"Filled NaN in {col} with mean: {fill_value:.2f}")


        print("\n--- Consolidated Historical Data with New Features (Sample) ---")
        print(final_df.head())
        print(f"\nTotal rows collected: {len(final_df)}")

        # Save the consolidated data to a CSV
        output_file = "f1_historical_data_with_features.csv" # New filename
        final_df.to_csv(output_file, index=False)
        print(f"\nConsolidated data saved to {output_file}")
    else:
        print("\nNo data collected for the specified years. Please check for errors during data loading.")

