# env_lstm.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class LSTMEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, model, scaler_input, scaler_output,
                 window_size=30, setpoint=5.0, max_steps=200, device="cpu"):
        super().__init__()
        self.model = model.eval().to(device)
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        self.window_size = window_size
        self.setpoint = setpoint
        self.max_steps = max_steps
        self.device = device

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(window_size, 2), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history = np.zeros((self.window_size, 2), dtype=np.float32)
        self.current_step = 0
        return self.history, {}

    def step(self, action):
        self.current_step += 1
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # scale input
        scaled_input = float(self.scaler_input.transform([[action_val]])[0][0])

        lstm_input = np.copy(self.history)
        lstm_input[-1, 0] = scaled_input
        lstm_input_tensor = torch.tensor(lstm_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred = self.model(lstm_input_tensor).cpu().numpy().flatten()[0]

        output_real = float(self.scaler_output.inverse_transform([[y_pred]])[0, 0])

        # update history
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1, :] = [action_val, output_real]

        # reward = negative MSE
        reward = -((output_real - self.setpoint) ** 2)

        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {"output": output_real}

        return self.history, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Step {self.current_step}: output={self.history[-1,1]:.4f}, input={self.history[-1,0]:.4f}")
