# train_sac.py
from stable_baselines3 import SAC
from pathlib import Path
import joblib, torch
from src.models.lstm_model import LSTMModel
from .env_test import LSTMEnv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCALER_FOLDER = BASE_DIR / "config/predict_model"
MODEL_FOLDER = BASE_DIR / "experiments/fine_tune_model"

# load scalers
scaler_input = joblib.load(SCALER_FOLDER / "scaler_input.pkl")
scaler_output = joblib.load(SCALER_FOLDER / "scaler_output.pkl")

# load pretrained LSTM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_files = list(MODEL_FOLDER.glob("*.pth"))
latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
model = LSTMModel(input_dim=2, hidden_dim=50, layer_dim=1, output_dim=1).to(device)
model.load_state_dict(torch.load(latest_model_file, map_location=device))

# env
env = LSTMEnv(model, scaler_input, scaler_output, window_size=30, setpoint=5.0, max_steps=200, device=device)

# train SAC
agent = SAC("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=20000)  # train 20k steps

# save
SAVE_PATH = BASE_DIR / "experiments/sac_lstm_agent"
agent.save(SAVE_PATH)
print(f"✅ Agent saved at {SAVE_PATH}")
