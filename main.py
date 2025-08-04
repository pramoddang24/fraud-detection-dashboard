import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from threading import Thread, Event
import random
from collections import deque


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
    df = df.dropna(subset=['arm_id']).reset_index(drop=True)
    df['arm_id'] = df['arm_id'].astype(int)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    print("Simulating from provided CSV file.")
except FileNotFoundError:
    print("creditcard_2023.csv not found. Simulating data with a synthetic dataset.")
    data = {
        **{f"V{i}": np.random.randn(500) for i in range(1, 29)},
        'Amount': np.random.uniform(50, 25000, 500),
        'Class': [0] * 450 + [1] * 50
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
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins='*')

# Threading for simulation
thread = None
thread_stop_event = Event()
current_model = "thompson_sampling"

# Sliding window for Precision@100
sliding_window = deque(maxlen=100)


def precision_at_k(window, k=100):
    if len(window) < k:
        return 0.0
    top_k = sorted(window, key=lambda x: x['prob'], reverse=True)[:k]
    true_positives = sum(1 for item in top_k if item['is_fraud'])
    return true_positives / k if k > 0 else 0.0


# --- Simulation function ---
def background_stream():
    global current_model
    transaction_step = 0
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
        transaction_id = transaction_data.get('id', transaction_step)

        ts_arm = thompson_agent.select_arm()
        thompson_agent.update(ts_arm, transaction_class)
        ts_prob = thompson_agent.alpha[ts_arm] / (thompson_agent.alpha[ts_arm] + thompson_agent.beta[ts_arm])

        ucb_arm = ucb1_agent.select_arm()
        ucb1_agent.update(ucb_arm, transaction_class)
        ucb_prob = ucb1_agent.values[ucb_arm]

        lr_features = np.append(transaction_features, transaction_data['arm_id']).reshape(1, -1)
        lr_prob = logistic_regression.predict_proba(lr_features)[0][1]

        sliding_window.append({
            'ts_prob': ts_prob,
            'lr_prob': lr_prob,
            'is_fraud': transaction_class
        })

        is_fraud = False
        model_name = ""
        probability = 0.0

        if current_model == "thompson_sampling":
            is_fraud = ts_prob > 0.5 or transaction_class == 1
            model_name = "Thompson Sampling"
            probability = ts_prob
        elif current_model == "ucb1":
            is_fraud = ucb_prob > 0.5 or transaction_class == 1
            model_name = "UCB1"
            probability = ucb_prob
        elif current_model == "logistic_regression":
            is_fraud = lr_prob > 0.5 or transaction_class == 1
            model_name = "Logistic Regression"
            probability = lr_prob

        if str(cluster_id) in cluster_counts:
            cluster_counts[str(cluster_id)] += 1
        else:
            cluster_counts[str(cluster_id)] = 1

        ts_probs_window = [item['ts_prob'] for item in sliding_window]
        lr_probs_window = [item['lr_prob'] for item in sliding_window]
        true_labels_window = [item['is_fraud'] for item in sliding_window]

        ts_precision_at_100 = precision_at_k(
            [{'prob': p, 'is_fraud': l} for p, l in zip(ts_probs_window, true_labels_window)], k=100
        )
        lr_precision_at_100 = precision_at_k(
            [{'prob': p, 'is_fraud': l} for p, l in zip(lr_probs_window, true_labels_window)], k=100
        )

        data_to_emit = {
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
            'cluster_distribution': cluster_counts,
            'precision_at_100': {
                'ts': ts_precision_at_100,
                'lr': lr_precision_at_100
            }
        }

        for connection in websocket_connections:
            emit('streaming_data', data_to_emit, room=connection.sid)  # Use rooms for a single user

        transaction_step += 1
        time.sleep(0.5)


# --- Flask routes ---
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')


# --- SocketIO events ---
@socketio.on('connect')
def test_connect():
    print('Client connected')


@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


@socketio.on('start_streaming')
def start_streaming(data):
    """Starts the background thread for data streaming."""
    global thread, current_model
    print('Received start_streaming event.')
    current_model = data['model']
    if thread is None or not thread.is_alive():
        thread_stop_event.clear()
        thread = Thread(target=background_stream)
        thread.daemon = True
        thread.start()
        emit('status', 'running', room=request.sid)


@socketio.on('stop_streaming')
def stop_streaming():
    """Stops the background thread for data streaming."""
    global thread
    print('Received stop_streaming event.')
    if thread is not None and thread.is_alive():
        thread_stop_event.set()
        emit('status', 'stopped', room=request.sid)


@socketio.on('set_model')
def set_model(data):
    """Updates the model used for real-time predictions."""
    global current_model
    current_model = data['model']
    print(f"Active model set to: {current_model}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, debug=False, host='0.0.0.0', port=port)