import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from threading import Thread, Event
import random


# Mocking the MAB classes from your ipynb, which are required for unpickling
# the .pkl model files correctly.
class UCB1:
    def __init__(self, n_arms, alpha=2.0):
        self.n_arms = n_arms
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.cumulative_regret = 0.0

    def select_arm(self):
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        return np.argmax(self.values + np.sqrt(self.alpha * np.log(self.t) / (self.counts + 1e-6)))

    def update(self, arm, reward, optimal_reward=1):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward
        self.cumulative_regret += (optimal_reward - reward)


class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.cumulative_regret = 0.0

    def select_arm(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))

    def update(self, arm, reward, optimal_reward=1):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        self.cumulative_regret += (optimal_reward - reward)


# --- Load Models and Data ---
model_dir = "models"
try:
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    clusterer = joblib.load(os.path.join(model_dir, "cluster_model.pkl"))
    logistic_regression = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
    ucb1_agent = joblib.load(os.path.join(model_dir, "ucb1_agent.pkl"))
    thompson_agent = joblib.load(os.path.join(model_dir, "thompson_agent.pkl"))
    with open(os.path.join(model_dir, "arm_map.json")) as f:
        arm_map_data = json.load(f)
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: {e}. Please ensure the 'models' folder exists and contains all the required files.")
    exit()

# Extract mappings from arm_map.json
arm_to_id = arm_map_data.get("arm_to_id", {})
id_to_arm = {int(k): v for k, v in arm_map_data.get("id_to_arm", {}).items()}
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
n_arms = len(arm_to_id)

# Create a sample DataFrame to simulate transactions if creditcard_2023.csv is not available
try:
    df = pd.read_csv("creditcard_2023.csv")
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Amount_Quartile'] = pd.qcut(df['Amount'], 4, labels=False, duplicates='drop')
    df['Category'] = clusterer.predict(preprocessor.transform(df[features]))
    df['arm'] = df['Amount_Quartile'].astype(str) + '_' + df['Category'].astype(str)
    df['arm_id'] = df['arm'].map(arm_to_id)
    # Filter out any rows that don't have a valid arm_id
    df = df.dropna(subset=['arm_id']).reset_index(drop=True)
    df['arm_id'] = df['arm_id'].astype(int)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    print("Simulating from provided CSV file.")
except FileNotFoundError:
    print("creditcard_2023.csv not found. Simulating data with a synthetic dataset.")
    data = {
        **{f"V{i}": np.random.randn(500) for i in range(1, 29)},
        'Amount': np.random.uniform(50, 25000, 500),
        'Class': [0] * 450 + [1] * 50  # 10% fraud rate for simulation
    }
    df = pd.DataFrame(data)
    df['Log_Amount'] = np.log1p(df['Amount'])
    df['Amount_Quartile'] = pd.qcut(df['Amount'], 4, labels=False, duplicates='drop')
    df['Category'] = clusterer.predict(preprocessor.transform(df[features]))
    df['arm'] = df['Amount_Quartile'].astype(str) + '_' + df['Category'].astype(str)
    df['arm_id'] = df['arm'].map(arm_to_id)
    df = df.dropna(subset=['arm_id']).reset_index(drop=True)
    df['arm_id'] = df['arm_id'].astype(int)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)

# --- Flask and SocketIO setup ---
# --- Flask and SocketIO setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Change 'gevent' to 'eventlet' for compatibility with Render.com
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins='*')

# Threading for simulation
thread = None
thread_stop_event = Event()


# --- Simulation function ---
def background_stream():
    transaction_step = 0
    # The keys need to be strings for the JSON payload
    cluster_counts = {str(i): 0 for i in range(1, clusterer.n_clusters + 1)}

    while not thread_stop_event.is_set():
        if transaction_step >= len(shuffled_df):
            print("End of simulation data. Looping.")
            transaction_step = 0
            shuffled_df = df.sample(frac=1).reset_index(drop=True)
            time.sleep(2)

        transaction_data = shuffled_df.iloc[transaction_step]
        transaction_features = transaction_data[features].values.reshape(1, -1)
        transaction_class = int(transaction_data['Class'])

        # Preprocess and cluster the data
        scaled_features = preprocessor.transform(transaction_features)
        cluster_id = clusterer.predict(scaled_features)[0] + 1

        # Get predictions from MABs
        ts_arm = thompson_agent.select_arm()
        ucb_arm = ucb1_agent.select_arm()

        # Update MAB agents based on transaction outcome
        # Note: We're using the actual transaction class as a reward signal
        thompson_agent.update(ts_arm, transaction_class)
        ucb1_agent.update(ucb_arm, transaction_class)

        # Use the Logistic Regression model for a probability prediction
        lr_prob = \
        logistic_regression.predict_proba(np.append(transaction_features, transaction_data['arm_id']).reshape(1, -1))[
            0][1]

        # A simplified fraud alert logic based on a threshold
        is_fraud = lr_prob > 0.5 or transaction_class == 1
        model_name = "Logistic Regression"
        probability = lr_prob

        # Update cluster distribution counts
        if str(cluster_id) in cluster_counts:
            cluster_counts[str(cluster_id)] += 1
        else:
            cluster_counts[str(cluster_id)] = 1

        # Prepare and emit data to the frontend
        emit('streaming_data', {
            'transaction': {
                'timestamp': time.strftime("%H:%M:%S", time.localtime()),
                'amount': transaction_data['Amount']
            },
            'prediction': {
                'is_fraud': is_fraud,
                'model_name': model_name,
                'probability': probability,
                'cluster_id': cluster_id
            },
            'mab_performance': {
                'ts_regret': thompson_agent.cumulative_regret,
                'ucb_regret': ucb1_agent.cumulative_regret
            },
            'cluster_distribution': cluster_counts
        }, namespace='/', broadcast=True)

        transaction_step += 1
        time.sleep(0.5)


# --- Flask routes ---
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')


# --- SocketIO events ---
@socketio.on('connect')
def test_connect():
    global thread
    print('Client connected')
    if thread is None or not thread.is_alive():
        thread_stop_event.clear()
        thread = Thread(target=background_stream)
        thread.daemon = True
        thread.start()


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
    thread_stop_event.set()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)
