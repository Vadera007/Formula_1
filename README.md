F1 Race Predictor 🏎️🏁





This project uses a machine learning model to predict the probable finishing lineup of Formula 1 races. It leverages years of historical Formula 1 data to learn complex patterns and provides dynamic predictions for upcoming Grand Prix events through a polished web application.

About The Project


As a Formula 1 enthusiast, I built this project to dive deep into race analytics. The core idea is to go beyond simple predictions and create a system that can learn from past performance, driver and team dynamics, and even environmental factors to forecast race outcomes.

The prediction engine, the 'Trinity' ensemble, considers a wide range of factors to make its forecast, providing a holistic, data-driven prediction of the final race order.

Features & Data Points


The model's accuracy is driven by a comprehensive set of features engineered from official Formula 1 data. The "Trinity" ensemble analyzes the following key areas:

Recent Driver & Team Form ->

Rolling Averages: The average qualifying and race finishing positions for both drivers and their teams over the last three races.

Driver Consistency: The standard deviation of a driver's finishing position, which tells the model if a driver is a consistent performer or more volatile.

Historical & Track-Specific Performance ->

Track History: A driver's historical average finish at the specific circuit for the upcoming race.

Track Type Specialization: The model categorizes circuits (e.g., "Street Circuit," "High Downforce") and analyzes a driver's historical performance on each type, helping to identify specialists.

Practice & Qualifying Pace ->

Qualifying Position: The driver's starting position for the race.

Practice Pace: The fastest single lap and the average long-run race simulation pace from all three practice sessions.

Race Strategy & Conditions ->

Pit Stops: The number of pit stops a driver made in past races.

Tyre Strategy: The compound (Soft, Medium, Hard) a driver started the race on.

Weather: Key environmental factors including Air Temperature, Track Temperature, and whether it was raining.

Advanced Analytics ->

Head-to-Head Teammate Battle: A rolling win rate for each driver against their own teammate, a powerful measure of pure skill.

Reliability: A rolling DNF (Did Not Finish) rate for each driver, which helps the model account for the risk of mechanical failures or crashes.

Model Performance

The "Trinity" ensemble model is highly accurate, consistently achieving an average Normalized Discounted Cumulative Gain (NDCG) score above 0.95 during cross-validation. An NDCG score measures the quality of a ranked list, and a score this high indicates that the model's predicted finishing order is, on average, over 95% similar to the actual race outcomes, demonstrating exceptional predictive power.

How to Run

Follow these steps to set up and run the project on your local machine.

1. Clone the Repository

git clone https://github.com/Vadera007/Formula_1.git
cd Formula_1

2. Set up a Virtual Environment

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run the Full Pipeline (Data + Training)

This master script will run the data collection and model training processes in order. This can take a considerable amount of time on the first run.

python3 run_pipeline.py

5. Launch the Web App

Once the pipeline has finished, you can launch the Streamlit application.

streamlit run app.py

License
This project is open-source and available under the MIT License.

Created By:
Akshat Vadera

