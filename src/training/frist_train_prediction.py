import torch
import torch.nn as nn
import joblib
from pathlib import Path
import time

# กรุณาแก้ไข path และ import ให้ตรงกับโครงสร้างโปรเจคของคุณ
from src.models.lstm_model import LSTMModel
from src.data.generator import data_generator
from src.data.format_duration_time import format_duration

# กำหนดค่าพารามิเตอร์และ path ต่าง ๆ
SCALER_FOLDER = r"D:\Project_end\mainproject_fix\main_project\config\predict_model"
DATA_FOLDER = r"D:\Project_end\mainproject_fix\main_project\data\raw"
PATH_SAVE_MODEL = r"D:\Project_end\mainproject_fix\main_project\experiments\model_ex01"

WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10


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

