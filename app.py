# JAI SHREE RAM //
# JAI MATA DI //
# JAI HANUMAN JI //
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import pytz
import fastf1
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for the best possible dynamic design ---
st.markdown("""
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

/* Main Background - Radial Gradient for depth */
body, .stApp {
    background: radial-gradient(circle at 50% 100%, #151535, #0a0a1a);
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}

/* --- Main content wrapper --- */
.main-content {
    padding: 0 5vw;
    max-width: 1200px;
    margin: auto;
}

/* Main Title and Hero Section */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 4em;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
    text-shadow: 0 0 10px #00c6ff, 0 0 20px #00c6ff, 0 0 30px #00c6ff;
    letter-spacing: 2px;
    padding-top: 60px;
    margin-bottom: 0;
    animation: fadeIn 2s ease-in-out;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.2em;
    text-align: center;
    color: #a0a0ff;
    margin-top: 10px;
    margin-bottom: 60px;
    animation: fadeIn 2.5s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Section Header Divider */
.st-emotion-cache-10n2b1a { /* Target the Streamlit divider element */
    border-top: 2px solid #00c6ff;
    margin-bottom: 2rem;
    box-shadow: 0 0 10px #00c6ff;
    animation: pulse 4s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 10px #00c6ff; }
    50% { box-shadow: 0 0 20px #00c6ff; }
    100% { box-shadow: 0 0 10px #00c6ff; }
}

/* Podium Card Styling */
.podium-card {
    background: rgba(42, 42, 94, 0.7);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    border: 2px solid transparent;
    margin-bottom: 20px;
    position: relative;
    transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease;
    transform-style: preserve-3d;
}

/* Individual podium card animations and colors */
.p1:hover { transform: scale(1.1) translateY(-10px); box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4); border-color: #FFD700; }
.p2:hover { transform: translateX(-15px) scale(1.05); box-shadow: 0 10px 30px rgba(192, 192, 192, 0.3); border-color: #C0C0C0; }
.p3:hover { transform: translateX(15px) scale(1.05); box-shadow: 0 10px 30px rgba(205, 127, 50, 0.3); border-color: #CD7F32; }

.p1 { border-image: linear-gradient(45deg, #FFD700, #F0C000) 1; margin-top: 0; }
.p2 { border-image: linear-gradient(45deg, #C0C0C0, #A9A9A9) 1; margin-top: 50px; }
.p3 { border-image: linear-gradient(45deg, #CD7F32, #B87333) 1; margin-top: 50px; }

.trophy-icon { font-size: 2.5em; margin-bottom: 10px; }
.podium-rank { font-family: 'Orbitron', sans-serif; font-size: 2.5em; font-weight: 700; margin-bottom: 10px; }
.p1 .podium-rank { color: #FFD700; }
.p2 .podium-rank { color: #C0C0C0; }
.p3 .podium-rank { color: #CD7F32; }
.podium-card .driver-name { font-family: 'Inter', sans-serif; font-size: 1.4em; font-weight: 600; color: #ffffff !important; }
.podium-card .team-name { font-family: 'Inter', sans-serif; font-size: 1.0em; color: #cccccc !important; }

/* Full Grid Table Styling */
.stDataFrame {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 0 10px rgba(0, 198, 255, 0.2);
}
.stDataFrame th {
    background-color: rgba(0, 0, 0, 0.3);
    color: #ffffff;
    font-weight: 600;
    font-family: 'Orbitron', sans-serif;
    border-bottom: 2px solid #00c6ff;
}
.stDataFrame tr:hover {
    background-color: rgba(0, 198, 255, 0.1);
}

/* Footer Styling - Integrated and subtle */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to top, rgba(13, 13, 43, 0.8), transparent);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    color: #aaa;
    text-align: center;
    padding: 20px 0;
    font-size: 14px;
    z-index: 1000;
}

/* Hide the Streamlit three-dots menu */
.st-emotion-cache-s1h2w9 {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --- Configuration & Feature Lists ---
MODEL_FILE = "f1_ranker_model.joblib"
HISTORICAL_DATA_FILE = "f1_historical_data_with_features.csv"
PREDICTION_CACHE_DIR = 'cache_prediction'
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

# --- Helper Functions ---
@st.cache_data
def get_next_race_info():
    try:
        if not os.path.exists(PREDICTION_CACHE_DIR): os.makedirs(PREDICTION_CACHE_DIR)
        fastf1.Cache.enable_cache(PREDICTION_CACHE_DIR)
        schedule = fastf1.get_event_schedule(datetime.now().year, include_testing=False)
        schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize(pytz.utc)
        now_utc = pd.Timestamp.now(tz='UTC')
        upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')
        if not upcoming_races.empty: return upcoming_races.iloc[0]
    except Exception as e:
        st.error(f"Could not fetch next race info from FastF1: {e}")
    return None

# --- Main App Content Wrapper ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Hero Section
st.markdown('<h1 class="hero-title">F1 Race Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Predicting the future of Formula 1 with machine learning.</p>', unsafe_allow_html=True)


# --- Load Model and Data ---
if not os.path.exists(MODEL_FILE) or not os.path.exists(HISTORICAL_DATA_FILE):
    st.warning("Model or data files not found. Please run `data_collection.py` and `model_training.py` first.")
    st.stop()
model_pipeline = joblib.load(MODEL_FILE)
df = pd.read_csv(HISTORICAL_DATA_FILE)
df['group_id'] = df.groupby(['Year', 'RoundNumber']).ngroup()

# --- Display Prediction ---
st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)
next_race = get_next_race_info()

if next_race is not None:
    st.header(f"Prediction for the {next_race['EventName']} {next_race['EventDate'].year}", divider='rainbow')
    with st.spinner("Firing up the prediction engine..."):
        time.sleep(1) # Simulate a short delay for dramatic effect
        last_race_mask = df['group_id'] == df['group_id'].max()
        prediction_input_df = df[last_race_mask].copy()
        
        # --- Handle missing categorical features ---
        # Get all unique categories from the training data
        all_drivers = df['Driver'].unique()
        all_teams = df['Team'].unique()
        all_tyres = df['StartingTyreCompound'].unique()
        
        # Function to ensure consistent columns for prediction
        def create_dummy_columns(df_to_process, feature, all_categories):
            for cat in all_categories:
                df_to_process[f'{feature}_{cat}'] = (df_to_process[feature] == cat).astype(int)
            return df_to_process.drop(columns=[feature])

        # Pre-process the input data for the model
        # The model pipeline handles this, but it's good practice to ensure
        # the input DataFrame has the same structure as the training data.
        # This is typically handled by a scikit-learn pipeline, but if not, this is a fix.
        
        predicted_scores = model_pipeline.predict(prediction_input_df[numerical_features + categorical_features])
        prediction_input_df['PredictedScore'] = predicted_scores
        predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
        predicted_lineup_df['Predicted Position'] = predicted_lineup_df.index + 1

    # Podium Section
    st.subheader("Predicted Podium")
    podium = predicted_lineup_df.head(3)
    if len(podium) == 3:
        p1, p2, p3 = podium.iloc[0], podium.iloc[1], podium.iloc[2]
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.markdown(f'<div class="podium-card p1"><div class="trophy-icon">🥇</div><div class="podium-rank">P{p1["Predicted Position"]}</div><div class="driver-name">{p1["FullName"]}</div><div class="team-name">{p1["Team"]}</div></div>', unsafe_allow_html=True)
        with col1:
            st.markdown(f'<div class="podium-card p2"><div class="trophy-icon">🥈</div><div class="podium-rank">P{p2["Predicted Position"]}</div><div class="driver-name">{p2["FullName"]}</div><div class="team-name">{p2["Team"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="podium-card p3"><div class="trophy-icon">🥉</div><div class="podium-rank">P{p3["Predicted Position"]}</div><div class="driver-name">{p3["FullName"]}</div><div class="team-name">{p3["Team"]}</div></div>', unsafe_allow_html=True)

    # Full Grid Section
    st.subheader("Full Predicted Grid")
    st.dataframe(
        predicted_lineup_df[['Predicted Position', 'FullName', 'Team']],
        use_container_width=True, hide_index=True
    )
else:
    st.info("No upcoming races found in the schedule.")

# --- About Section ---
st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.header("About This Project", divider='rainbow')
st.markdown("""
This project uses machine learning to predict the finishing order of Formula 1 races. It's built with a comprehensive data pipeline that fetches historical and live F1 data using the **FastF1** library.
The core of the predictor is an **XGBoost Ranking model**, which is specifically designed to learn the correct order of drivers rather than their exact finishing position.

The model is trained on a rich set of features, including:
- **Historical Performance:** Rolling averages and consistency metrics for drivers and teams.
- **Practice & Qualifying Pace:** Data from practice sessions and qualifying to gauge a car's potential.
- **Race Strategy:** Features like the number of pit stops and starting tyre compound.

The frontend is built with **Streamlit**, providing an interactive and dynamic way to view the predictions.
""")

# --- Close the main content wrapper BEFORE the footer ---
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
<p>© 2025 F1 Race Predictor | A Project by Akshat Vadera</p>
</div>
""", unsafe_allow_html=True)









# #JAI SHREE RAM //
# #JAI MATA DI //
# #JAI HANUMAN JI //
# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import joblib
# from datetime import datetime
# import pytz
# import fastf1
# import time
# import sqlite3

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="F1 Race Predictor",
#     page_icon="🏁",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # --- Function to load CSS from a file ---
# def load_css(file_name):
#     try:
#         with open(file_name) as f:
#             return f.read()
#     except FileNotFoundError:
#         st.error(f"Could not find the CSS file: {file_name}")
#         return ""

# # --- Custom CSS and JavaScript ---
# css = load_css("style.css")
# js = """
# <!-- Load Tone.js for sound generation -->
# <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.min.js"></script>
# <script>
#     window.addEventListener('load', function() {
#         const synth = new Tone.Synth({
#             oscillator: { type: "square" },
#             envelope: {
#                 attack: 0.005,
#                 decay: 0.1,
#                 sustain: 0,
#                 release: 0.05
#             }
#         }).toDestination();
        
#         const podiumCards = document.querySelectorAll('.podium-card');
#         podiumCards.forEach(card => {
#             card.addEventListener('mouseenter', () => {
#                 // Play a bright, short tone on hover
#                 synth.triggerAttackRelease("C5", "16n");
#             });
#         });
#     });
# </script>
# """


# st.markdown(f"<style>{css}</style>{js}", unsafe_allow_html=True)


# # --- Configuration & Feature Lists ---
# XGB_MODEL_FILE = "xgb_model.joblib"
# LGBM_MODEL_FILE = "lgbm_model.joblib"
# CATBOOST_MODEL_FILE = "catboost_model.joblib"
# DB_FILE = "f1_predictor.db"
# numerical_features = [
#     'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
#     'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
#     'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
#     'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
#     'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
#     'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
#     'PracticeFastestLap', 'AvgLongRunPace',
#     'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance'
# ]
# categorical_features = ['Driver', 'Team', 'StartingTyreCompound', 'TrackType']

# # --- Helper Functions ---
# @st.cache_data
# def get_next_race_info():
#     try:
#         schedule = fastf1.get_event_schedule(datetime.now().year, include_testing=False)
#         schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize(pytz.utc)
#         now_utc = pd.Timestamp.now(tz='UTC')
#         upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')
#         if not upcoming_races.empty: return upcoming_races.iloc[0]
#     except Exception as e:
#         st.error(f"Could not fetch next race info: {e}")
#     return None

# @st.cache_data
# def load_data():
#     model_files = [XGB_MODEL_FILE, LGBM_MODEL_FILE, CATBOOST_MODEL_FILE]
#     if not all(os.path.exists(f) for f in model_files) or not os.path.exists(DB_FILE):
#         return None, None, None, None
#     xgb_model = joblib.load(XGB_MODEL_FILE)
#     lgbm_model = joblib.load(LGBM_MODEL_FILE)
#     catboost_model = joblib.load(CATBOOST_MODEL_FILE)
#     conn = sqlite3.connect(DB_FILE)
#     data = pd.read_sql('SELECT * FROM historical_data', conn)
#     conn.close()
#     data['group_id'] = data.groupby(['Year', 'RoundNumber']).ngroup()
#     return xgb_model, lgbm_model, catboost_model, data

# # --- Main App Content Wrapper ---
# st.markdown('<div class="main-content">', unsafe_allow_html=True)
# st.markdown('<h1 class="hero-title">Formula 1 Predictor</h1>', unsafe_allow_html=True)
# st.markdown('<p class="hero-subtitle">Predicting the future of Formula 1 with TRINITY (Machine Learning)</p>', unsafe_allow_html=True)

# # --- Load Models and Data ---
# xgb_model, lgbm_model, catboost_model, df = load_data()
# if df is None:
#     st.warning("Model or data files not found. Please run `data_collection.py` and the ensemble `model_training.py` first.")
#     st.stop()

# # --- Display Prediction ---
# next_race = get_next_race_info()
# if next_race is not None:
#     st.header(f"Prediction for the {next_race['EventName']} {next_race['EventDate'].year}", divider='rainbow')
#     with st.spinner("Firing up the ensemble engine..."):
#         time.sleep(1)
#         last_race_mask = df['group_id'] == df['group_id'].max()
#         prediction_input_df = df[last_race_mask].copy()
        
#         all_model_features = numerical_features + categorical_features
#         for col in all_model_features:
#             if col not in prediction_input_df.columns:
#                 if col in numerical_features:
#                     prediction_input_df[col] = df[col].mean()
#                 else:
#                     prediction_input_df[col] = df[col].mode()[0]
        
#         xgb_scores = xgb_model.predict(prediction_input_df[all_model_features])
#         lgbm_scores = lgbm_model.predict(prediction_input_df[all_model_features])
#         cat_input = prediction_input_df[all_model_features].copy()
#         cat_input['group_id'] = prediction_input_df['group_id']
#         catboost_scores = catboost_model.predict(cat_input)
        
#         ensemble_scores = (xgb_scores + lgbm_scores + catboost_scores) / 3.0
        
#         prediction_input_df['PredictedScore'] = ensemble_scores
#         predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
#         predicted_lineup_df['Predicted Position'] = predicted_lineup_df.index + 1

#     # Podium Section
#     st.subheader("Predicted Podium")
#     podium = predicted_lineup_df.head(3)
#     if len(podium) == 3:
#         p1, p2, p3 = podium.iloc[0], podium.iloc[1], podium.iloc[2]
        
#         # Use a single container for a more controlled flexbox layout
#         st.markdown(f"""
#         <div class="podium-container">
#             <div class="podium-card p2">
#                 <div class="podium-rank">P{p2["Predicted Position"]}</div>
#                 <div class="driver-name">{p2["FullName"]}</div>
#                 <div class="team-name">{p2["Team"]}</div>
#             </div>
#             <div class="podium-card p1">
#                 <div class="podium-rank">P{p1["Predicted Position"]}</div>
#                 <div class="driver-name">{p1["FullName"]}</div>
#                 <div class="team-name">{p1["Team"]}</div>
#             </div>
#             <div class="podium-card p3">
#                 <div class="podium-rank">P{p3["Predicted Position"]}</div>
#                 <div class="driver-name">{p3["FullName"]}</div>
#                 <div class="team-name">{p3["Team"]}</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     # Full Grid Section
#     st.subheader("Full Predicted Grid")
#     st.dataframe(predicted_lineup_df[['Predicted Position', 'FullName', 'Team']], use_container_width=True, hide_index=True)
# else:
#     st.info("No upcoming races found in the schedule.")

# # --- About Section ---
# st.header("About This Project", divider='rainbow')
# st.markdown("""
# This predictor goes beyond the starting grid to forecast the most likely race outcome. It's powered by a machine learning system that analyzes years of historical Formula 1 data to uncover deep performance patterns.

# Our prediction engine, the **'Trinity' ensemble**, considers a wide range of factors to make its forecast, including:

# -   **Recent Driver & Team Form:** How have they performed in the last few races?
# -   **Historical Track Performance:** Which drivers and teams have historically excelled at this specific circuit?
# -   **Practice & Qualifying Pace:** Who showed the most potential during the practice sessions and in the battle for pole position?
# -   **Simulated Race Strategy:** Factoring in variables like the likely number of pit stops and tyre choices.

# By combining these insights, the model provides a holistic, data-driven prediction of the final race order.
# """)

# st.markdown('</div>', unsafe_allow_html=True)

# # --- Footer ---
# st.markdown("""
# <div class="footer">
# <p> Formula 1 Predictor | A Project by Akshat Vadera </p>
# </div>
# """, unsafe_allow_html=True)







# # JAI SHREE RAM //
# # JAI MATA DI //
# # JAI HANUMAN JI //
# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import joblib
# from datetime import datetime
# import pytz
# import fastf1
# import time
# import sqlite3

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="F1 Race Predictor",
#     page_icon="🏁",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # --- Custom CSS for F1.com Inspired Look ---
# st.markdown("""
# <style>
#     /* Import modern fonts from Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

#     /* Main App Background */
#     body, .stApp {
#         background: #0d0d0d;
#         color: #f0f0f0;
#         font-family: 'Inter', sans-serif;
#     }
#     .main-content {
#         padding: 0 5vw 60px 5vw;
#         max-width: 1200px;
#         margin: auto;
#     }

#     /* Hero Section */
#     .hero-title {
#         font-family: 'Bebas Neue', sans-serif;
#         font-size: 6em;
#         font-weight: 700;
#         text-align: center;
#         color: #ffffff;
#         text-shadow: 0 0 15px #e10600, 0 0 30px #e10600;
#         letter-spacing: 2px;
#         padding-top: 60px;
#         margin-bottom: 0;
#         animation: fadeIn 2s ease-in-out;
#     }
#     .hero-subtitle {
#         font-family: 'Inter', sans-serif;
#         font-size: 1.2em;
#         text-align: center;
#         color: #d1d1d1;
#         margin-top: 10px;
#         margin-bottom: 60px;
#         animation: fadeIn 2.5s ease-in-out;
#     }
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(20px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     /* Custom Header/Divider Style */
#     h2, .st-emotion-cache-10n2b1a {
#         font-family: 'Bebas Neue', sans-serif;
#         font-size: 2.5em;
#         color: #fff;
#         text-shadow: 0 0 5px #e10600;
#         border-top: 2px solid #e10600;
#         margin-bottom: 2rem;
#         box-shadow: 0 0 10px #e10600;
#         animation: pulse 4s infinite;
#         text-align: center;
#         padding-top: 20px;
#         text-transform: uppercase;
#         letter-spacing: 1.5px;
#     }
#     @keyframes pulse {
#         0% { box-shadow: 0 0 10px #e10600; }
#         50% { box-shadow: 0 0 20px #e10600; }
#         100% { box-shadow: 0 0 10px #e10600; }
#     }

#     /* Podium Card Styling */
#     .podium-container {
#         display: flex;
#         justify-content: center;
#         align-items: flex-end;
#         gap: 20px;
#         margin-bottom: 40px;
#         padding-top: 50px;
#     }
#     .podium-card {
#         background: #1a1a1a;
#         border-radius: 15px;
#         padding: 30px;
#         text-align: center;
#         transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease;
#         flex: 1;
#         position: relative;
#         border: 2px solid transparent;
#     }

#     /* Initial podium card positions */
#     .podium-card.p1 { transform: translateY(-50px); background: #2a2a2a; }
#     .podium-card.p2 { transform: translateY(-25px); background: #202020; }
#     .podium-card.p3 { transform: translateY(-25px); background: #202020; }

#     /* Hover effects with color-specific borders and glow */
#     .podium-card.p1:hover {
#         transform: translateY(-55px) scale(1.05);
#         border-color: #FFD700;
#         box-shadow: 0 0 20px #FFD700, 0 0 30px #FFD700, 0 15px 30px rgba(255, 215, 0, 0.4);
#     }
#     .podium-card.p2:hover {
#         transform: translateY(-30px) scale(1.05);
#         border-color: #C0C0C0;
#         box-shadow: 0 0 20px #C0C0C0, 0 0 30px #C0C0C0, 0 15px 30px rgba(192, 192, 192, 0.4);
#     }
#     .podium-card.p3:hover {
#         transform: translateY(-30px) scale(1.05);
#         border-color: #CD7F32;
#         box-shadow: 0 0 20px #CD7F32, 0 0 30px #CD7F32, 0 15px 30px rgba(205, 127, 50, 0.4);
#     }
    
#     .podium-rank {
#         font-family: 'Bebas Neue', sans-serif;
#         font-size: 3em;
#         font-weight: 700;
#         margin-bottom: 10px;
#     }
#     .podium-card.p1 .podium-rank { color: #FFD700; }
#     .podium-card.p2 .podium-rank { color: #C0C0C0; }
#     .podium-card.p3 .podium-rank { color: #CD7F32; }
#     .podium-card .driver-name {
#         font-family: 'Roboto Mono', monospace;
#         font-size: 1.4em;
#         font-weight: 700;
#         color: #ffffff;
#     }
#     .podium-card .team-name {
#         font-family: 'Inter', sans-serif;
#         font-size: 1.0em;
#         color: #a0a0a0;
#     }

#     /* Full Grid Table Styling */
#     .stDataFrame {
#         background-color: #1a1a1a;
#         border-radius: 10px;
#         border: 1px solid #333;
#         box-shadow: 0 0 10px rgba(225, 6, 0, 0.2);
#     }
#     .stDataFrame th {
#         background-color: #333;
#         color: #ffffff;
#         font-weight: 600;
#         font-family: 'Bebas Neue', sans-serif;
#         border-bottom: 2px solid #e10600;
#     }
#     .stDataFrame tr:hover {
#         background-color: #2a2a2a;
#     }

#     /* Footer Styling */
#     .footer {
#         position: static;
#         width: 100%;
#         background: #1a1a1a;
#         color: #777;
#         text-align: center;
#         padding: 20px 0;
#         font-size: 14px;
#         border-top: 1px solid #333;
#     }
# </style>
# <!-- Load Tone.js for sound generation -->
# <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.min.js"></script>
# <script>
#     window.addEventListener('load', function() {
#         const synth = new Tone.Synth({
#             oscillator: { type: "square" },
#             envelope: {
#                 attack: 0.005,
#                 decay: 0.1,
#                 sustain: 0,
#                 release: 0.05
#             }
#         }).toDestination();
        
#         const podiumCards = document.querySelectorAll('.podium-card');
#         podiumCards.forEach(card => {
#             card.addEventListener('mouseenter', () => {
#                 // Play a bright, short tone on hover
#                 synth.triggerAttackRelease("C5", "16n");
#             });
#         });
#     });
# </script>
# """, unsafe_allow_html=True)

# # --- Configuration & Feature Lists ---
# XGB_MODEL_FILE = "xgb_model.joblib"
# LGBM_MODEL_FILE = "lgbm_model.joblib"
# CATBOOST_MODEL_FILE = "catboost_model.joblib"
# DB_FILE = "f1_predictor.db"
# numerical_features = [
#     'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
#     'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
#     'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
#     'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
#     'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
#     'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
#     'PracticeFastestLap', 'AvgLongRunPace',
#     'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance'
# ]
# categorical_features = ['Driver', 'Team', 'StartingTyreCompound', 'TrackType']

# # --- Helper Functions ---
# @st.cache_data
# def get_next_race_info():
#     try:
#         schedule = fastf1.get_event_schedule(datetime.now().year, include_testing=False)
#         schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize(pytz.utc)
#         now_utc = pd.Timestamp.now(tz='UTC')
#         upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')
#         if not upcoming_races.empty: return upcoming_races.iloc[0]
#     except Exception as e:
#         st.error(f"Could not fetch next race info: {e}")
#     return None

# @st.cache_data
# def load_data():
#     model_files = [XGB_MODEL_FILE, LGBM_MODEL_FILE, CATBOOST_MODEL_FILE]
#     if not all(os.path.exists(f) for f in model_files) or not os.path.exists(DB_FILE):
#         return None, None, None, None
#     xgb_model = joblib.load(XGB_MODEL_FILE)
#     lgbm_model = joblib.load(LGBM_MODEL_FILE)
#     catboost_model = joblib.load(CATBOOST_MODEL_FILE)
#     conn = sqlite3.connect(DB_FILE)
#     data = pd.read_sql('SELECT * FROM historical_data', conn)
#     conn.close()
#     data['group_id'] = data.groupby(['Year', 'RoundNumber']).ngroup()
#     return xgb_model, lgbm_model, catboost_model, data

# # --- Main App Content Wrapper ---
# st.markdown('<div class="main-content">', unsafe_allow_html=True)
# st.markdown('<h1 class="hero-title">Formula 1 Predictor</h1>', unsafe_allow_html=True)
# st.markdown('<p class="hero-subtitle">Predicting the future of Formula 1 with TRINITY (Machine Learning)</p>', unsafe_allow_html=True)

# # --- Load Models and Data ---
# xgb_model, lgbm_model, catboost_model, df = load_data()
# if df is None:
#     st.warning("Model or data files not found. Please run `data_collection.py` and the ensemble `model_training.py` first.")
#     st.stop()

# # --- Display Prediction ---
# next_race = get_next_race_info()
# if next_race is not None:
#     st.header(f"Prediction for the {next_race['EventName']} {next_race['EventDate'].year}", divider='rainbow')
#     with st.spinner("Firing up the ensemble engine..."):
#         time.sleep(1)
#         last_race_mask = df['group_id'] == df['group_id'].max()
#         prediction_input_df = df[last_race_mask].copy()
        
#         all_model_features = numerical_features + categorical_features
#         for col in all_model_features:
#             if col not in prediction_input_df.columns:
#                 if col in numerical_features:
#                     prediction_input_df[col] = df[col].mean()
#                 else:
#                     prediction_input_df[col] = df[col].mode()[0]
        
#         xgb_scores = xgb_model.predict(prediction_input_df[all_model_features])
#         lgbm_scores = lgbm_model.predict(prediction_input_df[all_model_features])
#         cat_input = prediction_input_df[all_model_features].copy()
#         cat_input['group_id'] = prediction_input_df['group_id']
#         catboost_scores = catboost_model.predict(cat_input)
        
#         ensemble_scores = (xgb_scores + lgbm_scores + catboost_scores) / 3.0
        
#         prediction_input_df['PredictedScore'] = ensemble_scores
#         predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
#         predicted_lineup_df['Predicted Position'] = predicted_lineup_df.index + 1

#     # Podium Section
#     st.subheader("Predicted Podium")
#     podium = predicted_lineup_df.head(3)
#     if len(podium) == 3:
#         p1, p2, p3 = podium.iloc[0], podium.iloc[1], podium.iloc[2]
        
#         # Use a single container for a more controlled flexbox layout
#         st.markdown(f"""
#         <div class="podium-container">
#             <div class="podium-card p2">
#                 <div class="podium-rank">P{p2["Predicted Position"]}</div>
#                 <div class="driver-name">{p2["FullName"]}</div>
#                 <div class="team-name">{p2["Team"]}</div>
#             </div>
#             <div class="podium-card p1">
#                 <div class="podium-rank">P{p1["Predicted Position"]}</div>
#                 <div class="driver-name">{p1["FullName"]}</div>
#                 <div class="team-name">{p1["Team"]}</div>
#             </div>
#             <div class="podium-card p3">
#                 <div class="podium-rank">P{p3["Predicted Position"]}</div>
#                 <div class="driver-name">{p3["FullName"]}</div>
#                 <div class="team-name">{p3["Team"]}</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#     # Full Grid Section
#     st.subheader("Full Predicted Grid")
#     st.dataframe(predicted_lineup_df[['Predicted Position', 'FullName', 'Team']], use_container_width=True, hide_index=True)
# else:
#     st.info("No upcoming races found in the schedule.")

# # --- About Section ---
# st.header("About This Project", divider='rainbow')
# st.markdown("""
# This predictor goes beyond the starting grid to forecast the most likely race outcome. It's powered by a machine learning system that analyzes years of historical Formula 1 data to uncover deep performance patterns.

# Our prediction engine, the **'Trinity' ensemble**, considers a wide range of factors to make its forecast, including:

# -   **Recent Driver & Team Form:** How have they performed in the last few races?
# -   **Historical Track Performance:** Which drivers and teams have historically excelled at this specific circuit?
# -   **Practice & Qualifying Pace:** Who showed the most potential during the practice sessions and in the battle for pole position?
# -   **Simulated Race Strategy:** Factoring in variables like the likely number of pit stops and tyre choices.

# By combining these insights, the model provides a holistic, data-driven prediction of the final race order.
# """)

# st.markdown('</div>', unsafe_allow_html=True)

# # --- Footer ---
# st.markdown("""
# <div class="footer">
# <p> Formula 1 Predictor | A Project by Akshat Vadera </p>
# </div>
# """, unsafe_allow_html=True)
















# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import os
# # import joblib
# # from datetime import datetime
# # import pytz
# # import fastf1
# # import time
# # import sqlite3

# # # --- Page Configuration ---
# # st.set_page_config(
# #     page_title="F1 Race Predictor",
# #     page_icon="🏁",
# #     layout="wide",
# #     initial_sidebar_state="collapsed"
# # )

# # # --- Custom CSS ---
# # st.markdown("""
# # <style>
# # @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
# # body, .stApp {
# #     background: radial-gradient(circle at 50% 100%, #151535, #0a0a1a);
# #     color: #e0e0e0;
# #     font-family: 'Inter', sans-serif;
# # }
# # .main-content { padding: 0 5vw 60px 5vw; max-width: 1200px; margin: auto; }
# # .hero-title {
# #     font-family: 'Orbitron', sans-serif; font-size: 4em; font-weight: 700;
# #     text-align: center; color: #ffffff;
# #     text-shadow: 0 0 10px #00c6ff, 0 0 20px #00c6ff, 0 0 30px #00c6ff;
# #     letter-spacing: 2px; padding-top: 60px; margin-bottom: 0;
# #     animation: fadeIn 2s ease-in-out;
# # }
# # .hero-subtitle {
# #     font-family: 'Inter', sans-serif; font-size: 1.2em; text-align: center;
# #     color: #a0a0ff; margin-top: 10px; margin-bottom: 60px;
# #     animation: fadeIn 2.5s ease-in-out;
# # }
# # @keyframes fadeIn {
# #     from { opacity: 0; transform: translateY(20px); }
# #     to { opacity: 1; transform: translateY(0); }
# # }
# # .st-emotion-cache-10n2b1a {
# #     border-top: 2px solid #00c6ff; margin-bottom: 2rem;
# #     box-shadow: 0 0 10px #00c6ff; animation: pulse 4s infinite;
# # }
# # @keyframes pulse {
# #     0% { box-shadow: 0 0 10px #00c6ff; }
# #     50% { box-shadow: 0 0 20px #00c6ff; }
# #     100% { box-shadow: 0 0 10px #00c6ff; }
# # }
# # .podium-card {
# #     background: rgba(42, 42, 94, 0.7); border-radius: 15px; padding: 20px;
# #     text-align: center; backdrop-filter: blur(10px); border: 2px solid transparent;
# #     margin-bottom: 20px; position: relative;
# #     transition: transform 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease;
# #     transform-style: preserve-3d;
# # }
# # .p1:hover { transform: scale(1.1) translateY(-10px); box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4); border-color: #FFD700; }
# # .p2:hover { transform: translateX(-15px) scale(1.05); box-shadow: 0 10px 30px rgba(192, 192, 192, 0.3); border-color: #C0C0C0; }
# # .p3:hover { transform: translateX(15px) scale(1.05); box-shadow: 0 10px 30px rgba(205, 127, 50, 0.3); border-color: #CD7F32; }
# # .p1 { border-image: linear-gradient(45deg, #FFD700, #F0C000) 1; margin-top: 0; }
# # .p2 { border-image: linear-gradient(45deg, #C0C0C0, #A9A9A9) 1; margin-top: 50px; }
# # .p3 { border-image: linear-gradient(45deg, #CD7F32, #B87333) 1; margin-top: 50px; }
# # .trophy-icon { font-size: 2.5em; margin-bottom: 10px; }
# # .podium-rank { font-family: 'Orbitron', sans-serif; font-size: 2.5em; font-weight: 700; margin-bottom: 10px; }
# # .p1 .podium-rank { color: #FFD700; }
# # .p2 .podium-rank { color: #C0C0C0; }
# # .p3 .podium-rank { color: #CD7F32; }
# # .podium-card .driver-name { font-family: 'Inter', sans-serif; font-size: 1.4em; font-weight: 600; color: #ffffff !important; }
# # .podium-card .team-name { font-family: 'Inter', sans-serif; font-size: 1.0em; color: #cccccc !important; }
# # .stDataFrame {
# #     background-color: rgba(255, 255, 255, 0.05); border-radius: 10px;
# #     border: 1px solid rgba(255, 255, 255, 0.1);
# #     box-shadow: 0 0 10px rgba(0, 198, 255, 0.2);
# # }
# # .stDataFrame th {
# #     background-color: rgba(0, 0, 0, 0.3); color: #ffffff; font-weight: 600;
# #     font-family: 'Orbitron', sans-serif; border-bottom: 2px solid #00c6ff;
# # }
# # .stDataFrame tr:hover { background-color: rgba(0, 198, 255, 0.1); }
# # .footer {
# #     position: fixed; left: 0; bottom: 0; width: 100%;
# #     background: linear-gradient(to top, rgba(13, 13, 43, 0.8), transparent);
# #     backdrop-filter: blur(5px); color: #aaa; text-align: center;
# #     padding: 20px 0; font-size: 14px; z-index: 1000;
# # }
# # </style>
# # """, unsafe_allow_html=True)

# # # --- Configuration & Feature Lists ---
# # XGB_MODEL_FILE = "xgb_model.joblib"
# # LGBM_MODEL_FILE = "lgbm_model.joblib"
# # CATBOOST_MODEL_FILE = "catboost_model.joblib"
# # DB_FILE = "f1_predictor.db" # UPDATED: Database file name
# # numerical_features = [
# #     'QualiPosition', 'AvgLapTime', 'LapsCompleted', 'AirTemp', 'TrackTemp', 'Rainfall',
# #     'DriverAvgQualiPositionLast3Races', 'DriverAvgRaceFinishPositionLast3Races',
# #     'DriverStdDevRaceFinishPositionLast3Races', 'TeamAvgQualiPositionLast3Races',
# #     'TeamAvgRaceFinishPositionLast3Races', 'TrackAvgQualiPosition',
# #     'TrackAvgRaceFinishPosition', 'TrackLengthKm', 'NumCorners',
# #     'NumPitStops', 'AvgStintLength', 'MedianLapTime', 'StdDevLapTime', 'AvgSector2Pace',
# #     'PracticeFastestLap', 'AvgLongRunPace',
# #     'TeammateWinRate', 'DriverDNFRate', 'TrackTypePerformance'
# # ]
# # categorical_features = ['Driver', 'Team', 'StartingTyreCompound', 'TrackType']

# # # --- Helper Functions ---
# # @st.cache_data
# # def get_next_race_info():
# #     try:
# #         schedule = fastf1.get_event_schedule(datetime.now().year, include_testing=False)
# #         schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize(pytz.utc)
# #         now_utc = pd.Timestamp.now(tz='UTC')
# #         upcoming_races = schedule[schedule['EventDate'] > now_utc].sort_values(by='EventDate')
# #         if not upcoming_races.empty: return upcoming_races.iloc[0]
# #     except Exception as e:
# #         st.error(f"Could not fetch next race info: {e}")
# #     return None

# # @st.cache_data
# # def load_data():
# #     model_files = [XGB_MODEL_FILE, LGBM_MODEL_FILE, CATBOOST_MODEL_FILE]
# #     if not all(os.path.exists(f) for f in model_files) or not os.path.exists(DB_FILE):
# #         return None, None, None, None
# #     xgb_model = joblib.load(XGB_MODEL_FILE)
# #     lgbm_model = joblib.load(LGBM_MODEL_FILE)
# #     catboost_model = joblib.load(CATBOOST_MODEL_FILE)
# #     # --- UPDATED: Read from SQLite Database ---
# #     conn = sqlite3.connect(DB_FILE)
# #     data = pd.read_sql('SELECT * FROM historical_data', conn)
# #     conn.close()
# #     data['group_id'] = data.groupby(['Year', 'RoundNumber']).ngroup()
# #     return xgb_model, lgbm_model, catboost_model, data

# # # --- Main App Content Wrapper ---
# # st.markdown('<div class="main-content">', unsafe_allow_html=True)
# # st.markdown('<h1 class="hero-title">F1 Race Predictor</h1>', unsafe_allow_html=True)
# # st.markdown('<p class="hero-subtitle">Predicting the future of Formula 1 with machine learning.</p>', unsafe_allow_html=True)

# # # --- Load Models and Data ---
# # xgb_model, lgbm_model, catboost_model, df = load_data()
# # if df is None:
# #     st.warning("Model or data files not found. Please run `data_collection.py` and the ensemble `model_training.py` first.")
# #     st.stop()

# # # --- Display Prediction ---
# # next_race = get_next_race_info()
# # if next_race is not None:
# #     st.header(f"Prediction for the {next_race['EventName']} {next_race['EventDate'].year}", divider='rainbow')
# #     with st.spinner("Firing up the ensemble engine..."):
# #         time.sleep(1)
# #         last_race_mask = df['group_id'] == df['group_id'].max()
# #         prediction_input_df = df[last_race_mask].copy()
        
# #         all_model_features = numerical_features + categorical_features
# #         for col in all_model_features:
# #             if col not in prediction_input_df.columns:
# #                 if col in numerical_features:
# #                     prediction_input_df[col] = df[col].mean()
# #                 else:
# #                     prediction_input_df[col] = df[col].mode()[0]
        
# #         xgb_scores = xgb_model.predict(prediction_input_df[all_model_features])
# #         lgbm_scores = lgbm_model.predict(prediction_input_df[all_model_features])
# #         cat_input = prediction_input_df[all_model_features].copy()
# #         cat_input['group_id'] = prediction_input_df['group_id']
# #         catboost_scores = catboost_model.predict(cat_input)
        
# #         ensemble_scores = (xgb_scores + lgbm_scores + catboost_scores) / 3.0
        
# #         prediction_input_df['PredictedScore'] = ensemble_scores
# #         predicted_lineup_df = prediction_input_df.sort_values(by='PredictedScore', ascending=False).reset_index(drop=True)
# #         predicted_lineup_df['Predicted Position'] = predicted_lineup_df.index + 1

# #     # Podium Section
# #     st.subheader("Predicted Podium")
# #     podium = predicted_lineup_df.head(3)
# #     if len(podium) == 3:
# #         p1, p2, p3 = podium.iloc[0], podium.iloc[1], podium.iloc[2]
# #         col1, col2, col3 = st.columns([1, 1, 1])
# #         with col2:
# #             st.markdown(f'<div class="podium-card p1"><div class="trophy-icon">🥇</div><div class="podium-rank">P{p1["Predicted Position"]}</div><div class="driver-name">{p1["FullName"]}</div><div class="team-name">{p1["Team"]}</div></div>', unsafe_allow_html=True)
# #         with col1:
# #             st.markdown(f'<div class="podium-card p2"><div class="trophy-icon">🥈</div><div class="podium-rank">P{p2["Predicted Position"]}</div><div class="driver-name">{p2["FullName"]}</div><div class="team-name">{p2["Team"]}</div></div>', unsafe_allow_html=True)
# #         with col3:
# #             st.markdown(f'<div class="podium-card p3"><div class="trophy-icon">🥉</div><div class="podium-rank">P{p3["Predicted Position"]}</div><div class="driver-name">{p3["FullName"]}</div><div class="team-name">{p3["Team"]}</div></div>', unsafe_allow_html=True)

# #     # Full Grid Section
# #     st.subheader("Full Predicted Grid")
# #     st.dataframe(predicted_lineup_df[['Predicted Position', 'FullName', 'Team']], use_container_width=True, hide_index=True)
# # else:
# #     st.info("No upcoming races found in the schedule.")

# # # --- UPDATED: About Section ---
# # st.header("About This Project", divider='rainbow')
# # st.markdown("""
# # This predictor goes beyond the starting grid to forecast the most likely race outcome. It's powered by a machine learning system that analyzes years of historical Formula 1 data to uncover deep performance patterns.

# # Our prediction engine, the **'Trinity' ensemble**, considers a wide range of factors to make its forecast, including:

# # -   **Recent Driver & Team Form:** How have they performed in the last few races?
# # -   **Historical Track Performance:** Which drivers and teams have historically excelled at this specific circuit?
# # -   **Practice & Qualifying Pace:** Who showed the most potential during the practice sessions and in the battle for pole position?
# # -   **Simulated Race Strategy:** Factoring in variables like the likely number of pit stops and tyre choices.

# # By combining these insights, the model provides a holistic, data-driven prediction of the final race order.
# # """)

# # st.markdown('</div>', unsafe_allow_html=True)

# # # --- Footer ---
# # st.markdown("""
# # <div class="footer">
# # <p>A Project by Akshat Vadera</p>
# # </div>
# # """, unsafe_allow_html=True)
