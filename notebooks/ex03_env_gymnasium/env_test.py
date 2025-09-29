import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import joblib
from pathlib import Path
from stable_baselines3 import SAC
from src.models.lstm_model import LSTMModel  # ปรับ path ให้ตรงกับโปรเจกต์

# --- Relative paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # main_project
SCALER_FOLDER = BASE_DIR / "config/predict_model"
MODEL_FOLDER = BASE_DIR / "experiments/fine_tune_model"

# โหลด scalers
scaler_input_file = SCALER_FOLDER / "scaler_input.pkl"
scaler_output_file = SCALER_FOLDER / "scaler_output.pkl"

if not scaler_input_file.exists() or not scaler_output_file.exists():
    raise FileNotFoundError("⚠️ Scaler files not found, run fine-tune scaler first.")

scaler_input = joblib.load(scaler_input_file)
scaler_output = joblib.load(scaler_output_file)

# โหลด pretrained LSTM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_files = list(MODEL_FOLDER.glob("*.pth"))
if not model_files:
    raise FileNotFoundError("⚠️ No pretrained LSTM model found in MODEL_FOLDER")

latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
model = LSTMModel(input_dim=2, hidden_dim=50, layer_dim=1, output_dim=1).to(device)
model.load_state_dict(torch.load(latest_model_file, map_location=device))
model.eval()


# --- Custom Gym Environment ---
class LSTMEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, model, scaler_input, scaler_output, window_size=30, setpoint=1.0, max_steps=200, device="cpu"):
        super().__init__()
        self.model = model
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        self.window_size = window_size
        self.setpoint = setpoint
        self.max_steps = max_steps
        self.device = device

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 2), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = np.zeros((self.window_size, 2), dtype=np.float32)
        self.current_step = 0
        return self.history, {}

    def step(self, action):
        self.current_step += 1
        action_val = np.clip(action[0], -1.0, 1.0)

        # Normalize input
        scaled_input = self.scaler_input.transform([[action_val]])[0]

        lstm_input = np.copy(self.history)
        lstm_input[-1, 0] = scaled_input
        lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred = self.model(lstm_input_tensor).cpu().numpy().flatten()[0]

        output_real = self.scaler_output.inverse_transform([[y_pred]])[0, 0]

        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1, :] = [action_val, output_real]

        reward = -((output_real - self.setpoint) ** 2)
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"output": output_real}

        return self.history, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}: output={self.history[-1,1]:.4f}, input={self.history[-1,0]:.4f}")


# --- Create environment ---
env = LSTMEnv(model, scaler_input, scaler_output, window_size=30, setpoint=5.0, max_steps=200, device=device)

# --- RL Agent ---
agent = SAC("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=10000)
