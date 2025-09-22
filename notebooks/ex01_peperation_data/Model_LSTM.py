import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time

# --- Config ---
DATA_FOLDER = r"D:\Project_end\mainproject\data\raw"
SCALER_FOLDER = r"D:\Project_end\mainproject\experiments\config\predict_model"
PATH_SAVE_MODEL = r"D:\Project_end\mainproject\experiments\config\predict_model"

# Hyperparameters
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


# -----------------------------
# LSTM Model
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Stateless LSTM → hidden state reset ทุก batch
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # เอาเฉพาะ timestep สุดท้าย
        return out

# -----------------------------
# Sequence Creator
# -----------------------------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 1])  # ใช้คอลัมน์ output
    return np.array(X), np.array(y)

def format_duration(total_seconds):

    if total_seconds <= 0:
        return "less 1 second"

    days = int(total_seconds // 86400)
    remaining_seconds = total_seconds % 86400
    hours = int(remaining_seconds // 3600)
    remaining_seconds %= 3600
    minutes = int(remaining_seconds // 60)
    seconds = remaining_seconds % 60

    # สร้าง List เพื่อเก็บข้อความแต่ละส่วน
    parts = []
    if days > 0:
        parts.append(f"{days} day")
    if hours > 0:
        parts.append(f"{hours} hours")
    if minutes > 0:
        parts.append(f"{minutes} minutes")
    
    parts.append(f"{seconds:.2f} second")
    return ", ".join(parts)

# -----------------------------
# Data Generator (อ่านทีละไฟล์ → ทีละ batch)
# -----------------------------
def data_generator(file_list, scaler_in, scaler_out, window_size, batch_size):
    for file_path in file_list:
        df = pd.read_csv(file_path)
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values.astype(np.float32)

        # scaling
        scaled_input = scaler_in.transform(data[:, 0].reshape(-1, 1))
        scaled_output = scaler_out.transform(data[:, 1].reshape(-1, 1))
        scaled_data = np.hstack([scaled_input, scaled_output])

        # สร้าง sequence
        X, y = create_sequences(scaled_data, window_size)

        # สร้าง batch
        for i in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float()
            y_batch = torch.from_numpy(y[i:i+batch_size]).float().view(-1, 1)
            yield X_batch, y_batch


# -----------------------------
# Training Loop
# -----------------------------
def train_model():
    print("--- 🚀 เริ่มการฝึกสอนโมเดล ---")

    # โหลด Scaler
    print("  -> โหลด Scaler ...")
    scaler_input = joblib.load(Path(SCALER_FOLDER) / "scaler_input.pkl")
    scaler_output = joblib.load(Path(SCALER_FOLDER) / "scaler_output.pkl")
    print("  -> โหลด Scaler สำเร็จ!")

    # เตรียมโมเดล
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_dim=2,
                      hidden_dim=HIDDEN_SIZE,
                      layer_dim=NUM_LAYERS,
                      output_dim=1).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ไฟล์ CSV ทั้งหมด
    csv_files = list(Path(DATA_FOLDER).glob("*.csv"))
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        train_gen = data_generator(csv_files, scaler_input, scaler_output,
                                   WINDOW_SIZE, BATCH_SIZE)

        for X_batch, y_batch in train_gen:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.6f}")

    total_time = time.time() - start_time
    formatted_time = format_duration(total_time)
    print(f"✅ Training finished. Total time: {formatted_time}")

    # บันทึกโมเดล
    torch.save(model.state_dict(), Path(PATH_SAVE_MODEL) / "lstm_model.pth")
    print(f"📂 โมเดลถูกบันทึกที่ {Path(PATH_SAVE_MODEL) / 'lstm_model.pth'}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    train_model()
