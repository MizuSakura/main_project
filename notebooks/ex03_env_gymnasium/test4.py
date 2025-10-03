import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from pathlib import Path
from notebooks.ex03_env_gymnasium.env_test import LSTMEnv
from src.models.lstm_model import LSTMModel

# ===== PWM Generator =====
class SignalGenerator:
    def __init__(self, t_end=10, dt=0.01):
        self.t = np.arange(0, t_end, dt)

    def pwm(self, amplitude=1, freq=1, duty=0.5):
        T = 1 / freq
        return amplitude * ((self.t % T) < duty * T)

# ===== Test PWM LSTMEnv =====
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCALER_FOLDER = BASE_DIR / "config/predict_model"
#MODEL_FOLDER = BASE_DIR / "experiments/fine_tune_model"
MODEL_FOLDER = BASE_DIR / "experiments/model_ex01"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load scalers
scaler_input = joblib.load(SCALER_FOLDER / "scaler_input.pkl")
scaler_output = joblib.load(SCALER_FOLDER / "scaler_output.pkl")

# Load pretrained LSTM
model_files = list(MODEL_FOLDER.glob("*.pth"))
latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
model = LSTMModel(input_dim=2, hidden_dim=50, layer_dim=1, output_dim=1).to(device)
model.load_state_dict(torch.load(latest_model_file, map_location=device))

# LSTM Env
env = LSTMEnv(model, scaler_input, scaler_output,
              window_size=30, setpoint=5.0, max_steps=2000, device=device)

# PWM signal
TIME_SIMULATION = 200
DT = 0.01
sg = SignalGenerator(t_end=TIME_SIMULATION*DT, dt=DT)
pwm_signal = sg.pwm(amplitude=1, freq=1, duty=1)

# Simulation
obs, _ = env.reset()
DATA_OUTPUT, ACTION = [], []
VOLT_SUPPLY = 24

for u in pwm_signal:
    action = np.array([u * VOLT_SUPPLY], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # เอา output ปัจจุบัน timestep ล่าสุด -> plot 1 เส้น
    DATA_OUTPUT.append(obs[-1,1])  
    ACTION.append(action[0])
    


# Plot
plt.figure(figsize=(12,6))
plt.plot(sg.t[:len(DATA_OUTPUT)], DATA_OUTPUT, label="LSTM Output")
plt.plot(sg.t[:len(ACTION)], ACTION, label="PWM Input", alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("LSTMEnv Response to PWM Signal")
plt.legend()
plt.grid(True)
plt.show()
