Formula 1 Race Predictor 🏎️💨

A Detailed Project Document 

This document provides a comprehensive overview of the Formula 1 Race Prediction Model, detailing every step of its development from initial setup to the final, interactive web application and automated pipeline.

1. Project Goal
The primary objective is to create a high end, self-updating machine learning pipeline that accurately predicts the finishing order of Formula 1 races. The project culminates in a user-friendly web application that presents these dynamic, data driven predictions in a visually appealing and interactive format, complete with a timestamp indicating the last model update.

2. Technical Stack

The project is built entirely in Python and utilizes a specific set of libraries:

FastF1: The core library for fetching official historical and live F1 data.

Pandas: The foundational library for all data handling, cleaning, and feature engineering.

Scikit-learn: Used for building the model pipeline, preprocessing data, and evaluation.

XGBoost, LightGBM, CatBoost: The three top-tier ranking models that form the "Trinity" ensemble.

Streamlit: The web framework used to create the interactive user interface.

Joblib: For saving and loading the trained machine learning models.

SQLite: For robust, file-based database storage.


3. The Pipeline

The project follows a three-stage pipeline, with separate scripts for data collection, model training, and the web application, ensuring a clean and modular workflow.

Stage 1: Data Collection & Feature Engineering (data_collection.py)

This script is the engine of the project. It connects to the FastF1 API, downloads data for all seasons from 2018 to the present, and transforms that raw data into a rich set of features which are then stored in a local SQLite database (f1_predictor.db). 

Engineered Features: The model's accuracy is driven by these carefully crafted features, which provide a holistic view of performance:

Qualifying & Grid Position:

QualiPosition: The driver's starting position for the race.

Driver & Team Form (Rolling 3-Race Averages):

DriverAvgQualiPositionLast3Races: A driver's recent one-lap pace.

DriverAvgRaceFinishPositionLast3Races: A driver's recent race performance.

TeamAvgQualiPositionLast3Races: The recent one lap pace of the team as a whole.

TeamAvgRaceFinishPositionLast3Races: The recent race performance of the team.

Consistency & Reliability:

DriverStdDevRaceFinishPositionLast3Races: The standard deviation of a driver's finishing position, quantifying their consistency (lower is better).

DriverDNFRate: A rolling DNF (Did Not Finish) rate for each driver, accounting for reliability risk.

Head-to-Head Performance:

TeammateWinRate: A rolling win rate against a driver's own teammate, a powerful measure of pure skill.

Track & Circuit Characteristics:

TrackLengthKm: The length of the circuit in kilometers.

NumCorners: The approximate number of corners on the track.

TrackType: A classification of the circuit (e.g., "Street Circuit," "High Downforce," "Low Downforce").

TrackTypePerformance: A driver's historical average performance on that specific type of circuit.

TrackAvgRaceFinishPosition: The historical average finishing position for all drivers at a specific Grand Prix.

Pace & Performance Metrics:

PracticeFastestLap: The driver's single fastest lap across all practice sessions.

AvgLongRunPace: The driver's median pace during their best long race simulation in practice.

MedianLapTime: The driver's median lap time during the race, a robust measure of consistent pace.

StdDevLapTime: The consistency of a driver's lap times during the race.

AvgSector2Pace: Average time in the often performance-critical second sector of the lap.

Race Strategy & Conditions:

NumPitStops: The number of pit stops made.

StartingTyreCompound: The tyre compound (Soft, Medium, Hard) the driver started on.

AvgStintLength: The average number of laps a driver completed between pit stops.

AirTemp & TrackTemp: Air and track temperatures.

Rainfall: A binary indicator for wet weather conditions.



Stage 2: Model Training & Evaluation (model_training.py)

This script is the core of the machine learning pipeline. It loads the pre-processed data from the SQLite database, evaluates the models, trains the final ensemble, and saves the models for the web app.

The "Trinity" Ensemble Model:

Algorithm: The model is an ensemble of three powerful gradient boosting rankers: XGBoost, LightGBM, and CatBoost. Instead of predicting an exact position, these models are trained to learn the correct order of drivers. The final prediction is an average of their outputs, making it more robust and accurate than any single model.

Evaluation: The script performs a robust GroupKFold cross-validation to measure the performance of each individual model and the final ensemble. The model's accuracy is measured using the NDCG (Normalized Discounted Cumulative Gain) score, a standard metric for ranking quality.

Performance:
The "Trinity" ensemble model is highly accurate, consistently achieving an average NDCG score above 0.95 during cross-validation. This indicates that the model's predicted finishing order is, on average, over 95% similar to the actual race outcomes, demonstrating exceptional predictive power.



Stage 3: Interactive Frontend (app.py)

This script creates a dynamic and visually appealing web application using Streamlit. It loads the saved "Trinity" models and the database to display the prediction for the next upcoming race.

Features:

Dynamic Content: Automatically identifies and displays the name of the next race.

Visual Design: Features a modern, dark themed interface with a professional feel.

Interactive Podium: Highlights the top 3 predicted finishers in styled cards with interactive hover effects.

Full Grid View: Presents the complete predicted finishing order in a clean, readable table.

Last Updated Timestamp: Displays the exact date and time the models were last retrained, giving users confidence in the data's freshness.

4. Automation

The entire data and modeling pipeline is fully automated using GitHub Actions. The pipeline is scheduled to run every 6 hours. This process:

Check out the latest code.

Runs the data collection script, which efficiently fetches new race data.

Runs the model training script to retrain the "Trinity" ensemble on the newly updated dataset.

Commits the new database, models, and timestamp file back to the repository

This means the deployed website will always show the latest, most accurate predictions without any manual intervention.



How to Run the Project

Follow these steps to set up and run the Formula 1 Race Predictor on your local machine.

1. Clone the Repository

First, clone the project repository from GitHub to your local machine using the following command in your terminal:

Bash:

git clone https://github.com/Vadera007/Formula_1.git

Then, navigate into the newly created project directory:

Bash:

cd Formula_1

2. Set Up the Environment

It's highly recommended to use a virtual environment to manage project dependencies.

Create a virtual environment:

Bash:

Activate the environment:

python3 -m venv venv

On macOS/Linux:

Bash:

source venv/bin/activate

On Windows:

Bash:

.\venv\Scripts\activate


3. Install Dependencies

Once your virtual environment is active, install all the required Python libraries using the requirements.txt file:

Bash:

pip install -r requirements.txt

4. Run the Full Pipeline

Now, execute the master script to collect the latest data and retrain the "Trinity" models. This script runs both data_collection.py and model_training.py in sequence.

Bash:

python3 run_pipeline.py

Wait for this process to complete. It might take some time as years of data is currently getting downloaded. Then it will print status updates to your terminal.

5. Launch the Web App

Once the pipeline has finished successfully, you can launch the Streamlit application to view the latest predictions.

Bash:

streamlit run app.py

Your web browser should automatically open with the F1 Race Predictor interface.

Have a look 👀

📄 Full Detailed Project Document: https://docs.google.com/document/d/14w67atQS5IbDBdrSC80oS50eq0GWTu_8SsKxsEa9_-0/edit?tab=t.0

🌐 Live App: https://formula1-predictor.streamlit.app

License This project is open-source and available under the MIT License.

Created by: Akshat Vadera
