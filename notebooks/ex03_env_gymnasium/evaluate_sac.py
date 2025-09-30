# evaluate_sac.py
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import numpy as np
import joblib, torch
from pathlib import Path
from src.models.lstm_model import LSTMModel
from .env_test import LSTMEnv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCALER_FOLDER = BASE_DIR / "config/predict_model"
MODEL_FOLDER = BASE_DIR / "experiments/fine_tune_model"
AGENT_PATH = BASE_DIR / "experiments/sac_lstm_agent.zip"

# load scalers
scaler_input = joblib.load(SCALER_FOLDER / "scaler_input.pkl")
scaler_output = joblib.load(SCALER_FOLDER / "scaler_output.pkl")

# load LSTM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_files = list(MODEL_FOLDER.glob("*.pth"))
latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
model = LSTMModel(input_dim=2, hidden_dim=50, layer_dim=1, output_dim=1).to(device)
model.load_state_dict(torch.load(latest_model_file, map_location=device))

# env
env = LSTMEnv(model, scaler_input, scaler_output, window_size=30, setpoint=5.0,
              max_steps=500, device=device)  # run long horizon

# load agent
agent = SAC.load(AGENT_PATH, env=env, device=device)
print("✅ Agent loaded")

# simulate
obs, _ = env.reset()
outputs, actions = [], []

for step in range(env.max_steps):
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    outputs.append(info["output"])
    actions.append(action[0])

    if terminated:
        break

# plot
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(outputs, label="Output")
plt.axhline(env.setpoint, color="r", linestyle="--", label="Setpoint")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(actions, label="Action", color="orange")
plt.ylabel("Action")
plt.xlabel("Step")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
