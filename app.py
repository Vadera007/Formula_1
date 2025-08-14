import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import pytz
import fastf1
import time
import sqlite3

# --- Page Configuration ---
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Function to load CSS from a file ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Could not find the CSS file: {file_name}")
        return ""

# --- Custom CSS and JavaScript ---
css = load_css("style.css")
js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.min.js"></script>
<script>
    window.addEventListener('load', function() {
        const synth = new Tone.Synth({
            oscillator: { type: "square" },
            envelope: {
                attack: 0.005,
                decay: 0.1,
                sustain: 0,
                release: 0.05
            }
        }).toDestination();
        
        const podiumCards = document.querySelectorAll('.podium-card');
        podiumCards.forEach(card => {
            card.addEventListener('mouseenter', () => {
                // Play a bright, short tone on hover
                synth.triggerAttackRelease("C5", "16n");
            });
        });
    });
</script>
"""
st.markdown(f"<style>{css}</style>{js}", unsafe_allow_html=True)


# --- Configuration & Feature Lists ---
XGB_MODEL_FILE = "Trinity/xgb_model.joblib"
LGBM_MODEL_FILE = "Trinity/lgbm_model.joblib"
CATBOOST_MODEL_FILE = "Trinity/catboost_model.joblib"
DB_FILE = "f1_predictor.db"

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

# --- Helper Functions ---
@st.cache_data
def get_next_race_info():
    try:
        schedule = fastf1.get_event_schedule(datetime.now().year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize(pytz.utc)
        now_utc = pd.Timestamp.now(tz='UTC')
        upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')
        if not upcoming_races.empty:
            return upcoming_races.iloc[0]
    except Exception as e:
        st.error(f"Could not fetch next race info: {e}")
    return None

@st.cache_data
def load_data():
    """Loads models and data, with robust error handling."""
    model_files = {
        "XGBoost": XGB_MODEL_FILE,
        "LightGBM": LGBM_MODEL_FILE,
        "CatBoost": CATBOOST_MODEL_FILE
    }
    
    # Checks for file existence first
    if not all(os.path.exists(f) for f in model_files.values()) or not os.path.exists(DB_FILE):
        st.error("One or more required files are missing from the directory. Please ensure all model files and the database are present.")
        return None, None, None, None

    try:
        # Loads models
        xgb_model = joblib.load(model_files["XGBoost"])
        lgbm_model = joblib.load(model_files["LightGBM"])
        catboost_model = joblib.load(model_files["CatBoost"])

        # Connects and reads from the database
        conn = sqlite3.connect(DB_FILE)
        data = pd.read_sql('SELECT * FROM historical_data', conn)
        conn.close()

        # Checks if data was loaded
        if data.empty:
            st.error(f"The 'historical_data' table in {DB_FILE} is empty!")
            return None, None, None, None
            
        data['group_id'] = data.groupby(['Year', 'RoundNumber']).ngroup()
        return xgb_model, lgbm_model, catboost_model, data

    except Exception as e:
        # Catches any other error during loading
        st.error(f"An error occurred while loading the data or models: **{e}**")
        return None, None, None, None

def get_last_updated_time():
    """Reads the timestamp from the saved file."""
    try:
        with open("last_updated.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Not available"

# --- Main App Content ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Formula 1 Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Predicting the future of Formula 1 with TRINITY (Machine Learning)</p>', unsafe_allow_html=True)

# Loads Models and Data
xgb_model, lgbm_model, catboost_model, df = load_data()

if df is None:
    st.warning("Prediction models could not be loaded. Please check the error messages above and ensure the pipeline has run successfully.")
    st.stop()

# --- Display Prediction ---
next_race = get_next_race_info()
if next_race is not None:
    st.header(f"Prediction for the {next_race['EventName']} {next_race['EventDate'].year}", divider='rainbow')
    with st.spinner("Firing up the ensemble engine..."):
        time.sleep(1) # A short delay for dramatic effect
        last_race_mask = df['group_id'] == df['group_id'].max()
        prediction_input_df = df[last_race_mask].copy()
        
        all_model_features = numerical_features + categorical_features
        for col in all_model_features:
            if col not in prediction_input_df.columns:
                if col in numerical_features:
                    prediction_input_df[col] = df[col].mean()
                else:
                    prediction_input_df[col] = df[col].mode()[0]
        
        # Ensures all features are present before prediction
        X_pred = prediction_input_df[all_model_features]

        xgb_scores = xgb_model.predict(X_pred)
        lgbm_scores = lgbm_model.predict(X_pred)
        
        # Prepares data for CatBoost if it was trained with group_id
        try:
            catboost_scores = catboost_model.predict(X_pred)
        except Exception:
            X_pred_cat = X_pred.copy()
            X_pred_cat['group_id'] = prediction_input_df['group_id']
            catboost_scores = catboost_model.predict(X_pred_cat)

        
        ensemble_scores = (xgb_scores + lgbm_scores + catboost_scores) / 3.0
        
        prediction_input_df['PredictedScore'] = ensemble_scores
        predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
        predicted_lineup_df['Predicted Position'] = predicted_lineup_df.index + 1

    # Podium Section
    st.subheader("Predicted Podium")
    podium = predicted_lineup_df.head(3)
    if len(podium) == 3:
        p1, p2, p3 = podium.iloc[0], podium.iloc[1], podium.iloc[2]
        
        st.markdown(f"""
        <div class="podium-container">
            <div class="podium-card p2">
                <div class="podium-rank">P{p2["Predicted Position"]}</div>
                <div class="driver-name">{p2["FullName"]}</div>
                <div class="team-name">{p2["Team"]}</div>
            </div>
            <div class="podium-card p1">
                <div class="podium-rank">P{p1["Predicted Position"]}</div>
                <div class="driver-name">{p1["FullName"]}</div>
                <div class="team-name">{p1["Team"]}</div>
            </div>
            <div class="podium-card p3">
                <div class="podium-rank">P{p3["Predicted Position"]}</div>
                <div class="driver-name">{p3["FullName"]}</div>
                <div class="team-name">{p3["Team"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Full Grid Section
    st.subheader("Full Predicted Grid")
    st.dataframe(predicted_lineup_df[['Predicted Position', 'FullName', 'Team']], use_container_width=True, hide_index=True)
else:
    st.info("No upcoming races found in the schedule.")

# --- About Section ---
st.header("About This Project", divider='rainbow')
st.markdown("""
This predictor goes beyond the starting grid to forecast the most likely race outcome. It's powered by a machine learning system that analyzes years of historical Formula 1 data to uncover deep performance patterns.

Our prediction engine, the **'Trinity' ensemble**, considers a wide range of factors to make its forecast, including:

-   **Recent Driver & Team Form:** How have they performed in the last few races?
-   **Historical Track Performance:** Which drivers and teams have historically excelled at this specific circuit?
-   **Practice & Qualifying Pace:** Who showed the most potential during the practice sessions and in the battle for pole position?
-   **Simulated Race Strategy:** Factoring in variables like the likely number of pit stops and tyre choices.

By combining these insights, the model provides a holistic, data-driven prediction of the final race order.
""")

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
last_updated = get_last_updated_time()
st.markdown(f"""
<div class="footer">
<p> Formula 1 Predictor | A Project by Akshat Vadera </p>
<p style="font-size: 0.8em; color: #888;">Last Updated: {last_updated}</p>
</div>
""", unsafe_allow_html=True)
