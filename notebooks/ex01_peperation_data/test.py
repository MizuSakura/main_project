import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time

# --- การตั้งค่า ---
DATA_FOLDER = "raw"
SCALER_FOLDER = "scalers"
MODEL_FOLDER = "models"

# Hyperparameters
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # ลดจำนวนลงเพื่อการทดสอบที่รวดเร็ว

# --- ส่วนประกอบที่ 1: โครงสร้างโมเดล (LSTM/GRU) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- ส่วนประกอบที่ 2: ฟังก์ชันสร้าง Sequences ---
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 1])
    return np.array(X), np.array(y)

# --- ส่วนประกอบที่ 3: Generator ป้อนข้อมูล ---
def data_generator(file_list, scaler_in, scaler_out, window_size, batch_size):
    for file_path in file_list:
        df = pd.read_csv(file_path)
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values.astype(np.float32)
        
        scaled_input = scaler_in.transform(data[:, 0].reshape(-1, 1))
        scaled_output = scaler_out.transform(data[:, 1].reshape(-1, 1))
        scaled_data = np.hstack([scaled_input, scaled_output])
        
        X, y = create_sequences(scaled_data, window_size)
        
        for i in range(0, len(X), batch_size):
            X_batch = torch.from_numpy(X[i:i+batch_size]).float()
            y_batch = torch.from_numpy(y[i:i+batch_size]).float().view(-1, 1)
            yield X_batch, y_batch

# --- ฟังก์ชันหลักในการฝึกสอน ---
def train_model():
    print("--- 🚀 เฟส 2: เริ่มกระบวนการฝึกสอนโมเดล ---")
    
    # 1. โหลด Global Scaler ที่สร้างไว้
    print("  -> กำลังโหลด Scaler...")
    scaler_input = joblib.load(Path(SCALER_FOLDER) / "global_scaler_input.pkl")
    scaler_output = joblib.load(Path(SCALER_FOLDER) / "global_scaler_output.pkl")
    print("  -> โหลด Scaler สำเร็จ!")

    # 2. เตรียมโมเดล, Loss function, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=2, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. เริ่ม Training Loop
    csv_files = list(Path(DATA_FOLDER).glob("*.csv"))
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # สร้าง Generator ใหม่ทุก Epoch
        train_gen = data_generator(csv_files, scaler_input, scaler_output, WINDOW_SIZE, BATCH_SIZE)
        
        for X_batch, y_batch in train_gen:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch [{epoch+1:02d}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}')

    print(f"\n--- ✅ การฝึกสอนเสร็จสิ้น! ใช้เวลา: {time.time() - start_time:.2f} วินาที ---")
    
    # 4. บันทึกโมเดลที่ฝึกเสร็จแล้ว
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    model_path = Path(MODEL_FOLDER) / "lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 บันทึกโมเดลเรียบร้อยแล้วที่: {model_path}")

# --- สั่งให้โปรแกรมทำงาน ---
if __name__ == "__main__":
    train_model()

