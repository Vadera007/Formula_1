F1 Race Predictor 🏎️🏁

This project is a machine learning model designed to predict the probable finishing lineup of Formula 1 races. It leverages historical F1 data to learn complex patterns and provides dynamic predictions for upcoming Grand Prix events.

About the Project:
As a passionate F1 enthusiast, I built this project to dive deep into race analytics. The core idea is to go beyond simple predictions and create a system that can learn from past performance, driver and team dynamics, and even environmental factors to forecast race outcomes.

Key Features:

Automated Data Collection: Fetches comprehensive historical Formula 1 data (lap times, qualifying results, weather, etc.) directly from the FastF1 API for multiple seasons (2023, 2024, and completed 2025 races).

Advanced Feature Engineering: Calculates crucial historical performance metrics for each driver and team, such as average qualifying and race finish positions over their last three races.

Robust Machine Learning Model: Employs an XGBoost Regressor, a powerful gradient boosting algorithm, to learn complex, non-linear relationships within the data.

Hyperparameter Tuning: Utilizes GridSearchCV to automatically find the optimal settings for the XGBoost model, maximizing its predictive accuracy.

Dynamic Race Prediction: Identifies the next upcoming F1 race and generates a predicted finishing lineup based on the most up-to-date historical performance data available before that race.

Modular Design: Separates data collection and model training into distinct scripts for better organization, although the model training script can trigger data refresh.

Technologies Used:

Python 3.8+FastF1: For accessing F1 data.
Pandas: For data manipulation and analysis.
Scikit-learn: For machine learning utilities (model training, preprocessing, evaluation, GridSearchCV).
XGBoost: The core regression model.
Matplotlib: (Primarily for data exploration/visualization, though not explicitly used in final output scripts)
.pytz: For robust timezone handling in date comparisons.

Getting Started:

Follow these steps to set up and run the project on your local machine.

1. Clone the Repository: First, clone this GitHub repository to your local machine:git clone https://github.com/Vadera007/Formula_1.git and then cd Formula_1

2. Set up a Virtual EnvironmentIt's highly recommended to use a virtual environment to manage project dependencies.python -m venv .venv

3. Activate the Virtual EnvironmentOn Windows (Command Prompt/PowerShell):.venv\Scripts\activate On macOS/Linux (Bash/Zsh):source .venv/bin/activate

4. Install DependenciesWith your virtual environment activated, install all required libraries using the requirements.txt file:pip install -r requirements.txt

Note for macOS users: If you encounter an XGBoostError related to libomp.dylib, you might need to install OpenMP via Homebrew:brew install libomp

5. Data Cache Setup:
The project uses local cache directories (cache_main, cache_prediction) to store downloaded FastF1 data, speeding up subsequent runs. These directories will be created automatically when the scripts run.

How to Run the Project: 

The project is designed with two main scripts: data_collection.py and model_training.py. 

Collect and Prepare Historical Data: Run data_collection.py to fetch all historical F1 data, engineer the advanced features, and save it to f1_historical_data_with_features.csv. This step can take a considerable amount of time (tens of minutes to hours) on the first run as it downloads large datasets. Subsequent runs will be faster due to caching.python data_collection.py

Train Model and Get Predictions: Once data_collection.py has successfully completed and generated the f1_historical_data_with_features.csv file, run model_training.py. 
This script will: Load the pre-processed historical data. Train and tune the XGBoost model using GridSearchCV. Evaluate the model's performance on a test set. Attempt to predict the lineup for the next upcoming F1 race, using dynamically calculated historical performance metrics for the drivers and teams involved. python model_training.py

Important Note on Dynamic Prediction: For a truly realistic prediction of an upcoming race, the script attempts to fetch the qualifying session data for that future race. If the qualifying session hasn't happened yet (or its data isn't yet available via FastF1), the script will use a fallback: it will determine the driver lineup from the last race in your historical training data and apply a hypothetical 1-20 qualifying order. To get predictions based on actual qualifying results, run model_training.py after the qualifying session for the target race has concluded and its data is published.
Project Structure
f1_predictor/
├── .venv/                      # Python virtual environment (ignored by Git)
├── cache_main/                 # FastF1 main data cache (ignored by Git)
├── cache_prediction/           # FastF1 prediction data cache (ignored by Git)
├── f1_historical_data_with_features.csv  # Consolidated historical data (ignored by Git)
├── data_collection.py          # Script to collect and preprocess historical F1 data
├── model_training.py           # Script to train/tune the model and make predictions
├── requirements.txt            # Lists project dependencies
└── .gitignore                  # Specifies files/folders to ignore in Git

Feel free to fork this repository, explore the code, and suggest improvements!

License: This project is open-source and available under the MIT License.

Created By: Akshat Vadera
