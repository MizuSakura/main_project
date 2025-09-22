import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time
import os

# --- ส่วนนำเข้าจากสคริปต์เดิม ---
# (ในโปรเจกต์จริง ควรแยกคลาสและฟังก์ชันเหล่านี้ไว้ในไฟล์ utils.py)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 1])
    return np.array(X), np.array(y)

def data_generator(file_list, scaler_in, scaler_out, window_size, batch_size):
    for file_path in file_list:
        df = pd.read_csv(file_path)
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values.astype(np.float32)
        scaled_input = scaler_in.transform(data[:, 0].reshape(-1, 1))
        scaled_output = scaler_out.transform(data[:, 1].reshape(-1, 1))
        scaled_data = np.hstack([scaled_input, scaled_output])
        X, y = create_sequences(scaled_data, window_size)
        for i in range(0, len(X), batch_size):
            yield (torch.from_numpy(X[i:i+batch_size]).float(),
                   torch.from_numpy(y[i:i+batch_size]).float().view(-1, 1))

# --- การตั้งค่าสำหรับ Fine-Tuning ---
SCALER_FOLDER = "scalers"
MODEL_FOLDER = "models"
NEW_DATA_FOLDER = "new_week_data" # โฟลเดอร์สำหรับข้อมูลใหม่

# Hyperparameters สำหรับ Fine-Tuning
WINDOW_SIZE = 30
BATCH_SIZE = 64
FINE_TUNE_EPOCHS = 5 # Fine-tune ไม่ต้องใช้ Epoch เยอะ
FINE_TUNE_LR = 0.0001 # *** สำคัญมาก: ใช้ Learning Rate ที่ต่ำลง ***

# --- ฟังก์ชันหลักในการ Fine-Tune ---
def fine_tune_weekly_model():
    """
    ฟังก์ชันหลักที่ควบคุมกระบวนการ Fine-Tuning ทั้งหมด
    """
    print("--- 🚀 เริ่มกระบวนการ Fine-Tuning รายสัปดาห์ ---")
    
    # --- ขั้นตอนที่ 1: อัปเดต Scaler ด้วยข้อมูลใหม่ ---
    print("\n--- [ขั้นตอนที่ 1/2] กำลังอัปเดต Scaler... ---")
    try:
        # 1.1 โหลด Scaler เก่า
        input_scaler_path = Path(SCALER_FOLDER) / "global_scaler_input.pkl"
        output_scaler_path = Path(SCALER_FOLDER) / "global_scaler_output.pkl"
        scaler_input = joblib.load(input_scaler_path)
        scaler_output = joblib.load(output_scaler_path)
        print("  -> โหลด Scaler เดิมสำเร็จ")

        # 1.2 ค้นหาไฟล์ข้อมูลใหม่
        new_files = list(Path(NEW_DATA_FOLDER).glob("*.csv"))
        if not new_files:
            print(f"⚠️ ไม่พบข้อมูลใหม่ในโฟลเดอร์ '{NEW_DATA_FOLDER}', สิ้นสุดการทำงาน")
            return
        print(f"  -> พบข้อมูลใหม่ {len(new_files)} ไฟล์")

        # 1.3 ทำ partial_fit ด้วยข้อมูลใหม่
        for file_path in new_files:
            with pd.read_csv(file_path, chunksize=10000) as reader:
                for chunk in reader:
                    scaler_input.partial_fit(chunk[['DATA_INPUT']].values)
                    scaler_output.partial_fit(chunk[['DATA_OUTPUT']].values)
        print("  -> อัปเดต Scaler ด้วยข้อมูลใหม่เรียบร้อย")

        # 1.4 บันทึก Scaler ที่อัปเดตแล้วทับไฟล์เดิม
        joblib.dump(scaler_input, input_scaler_path)
        joblib.dump(scaler_output, output_scaler_path)
        print("  -> บันทึก Scaler ที่อัปเดตแล้วทับของเดิม")

    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์ Scaler เดิม! กรุณารันสคริปต์ 01_prepare_scaler.py ก่อน")
        return

    # --- ขั้นตอนที่ 2: โหลดโมเดลเก่าและฝึกต่อด้วยข้อมูลใหม่ ---
    print("\n--- [ขั้นตอนที่ 2/2] กำลัง Fine-Tune โมเดล... ---")
    try:
        # 2.1 เตรียมโมเดลและโหลด state จากสัปดาห์ที่แล้ว
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = Path(MODEL_FOLDER) / "lstm_model.pth"
        
        model = LSTMModel(input_size=2, hidden_size=50, num_layers=1, output_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  -> โหลดโมเดลจาก '{model_path}' สำเร็จ")

        # 2.2 ตั้งค่า Optimizer ด้วย Learning Rate ที่ต่ำลง
        optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        criterion = nn.MSELoss()
        
        # 2.3 เริ่ม Fine-Tuning Loop (ใช้เฉพาะข้อมูลใหม่)
        print(f"  -> เริ่มการ Fine-Tune ด้วย Learning Rate = {FINE_TUNE_LR}")
        start_time = time.time()
        model.train()
        
        for epoch in range(FINE_TUNE_EPOCHS):
            total_loss = 0
            num_batches = 0
            
            # Generator จะอ่านเฉพาะไฟล์ในโฟลเดอร์ข้อมูลใหม่
            fine_tune_gen = data_generator(new_files, scaler_input, scaler_output, WINDOW_SIZE, BATCH_SIZE)

            for X_batch, y_batch in fine_tune_gen:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f'    Epoch [{epoch+1:02d}/{FINE_TUNE_EPOCHS}], Loss: {avg_loss:.6f}')
        
        # 2.4 บันทึกโมเดลที่อัปเดตแล้วทับไฟล์เดิม
        torch.save(model.state_dict(), model_path)
        total_time = time.time() - start_time
        print(f"  -> บันทึกโมเดลที่อัปเดตแล้วทับของเดิมเรียบร้อย")
        print(f"\n✅ Fine-Tuning เสร็จสิ้น! ใช้เวลา: {total_time:.2f} วินาที")

    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์โมเดลเดิม! กรุณารันสคริปต์ 02_train_model.py ก่อน")
        return

# --- สั่งให้โปรแกรมทำงาน ---
if __name__ == "__main__":
    # --- จำลองสถานการณ์ ---
    # 1. สร้างโฟลเดอร์สำหรับข้อมูลใหม่ (ถ้ายังไม่มี)
    Path("new_week_data").mkdir(exist_ok=True)
    
    # 2. คัดลอกไฟล์ข้อมูลใดไฟล์หนึ่งมาเป็น "ข้อมูลใหม่"
    #    ในสถานการณ์จริง ไฟล์นี้จะมาจากระบบเก็บข้อมูลของคุณ
    #    (ตรวจสอบให้แน่ใจว่าคุณมีไฟล์นี้อยู่ในโฟลเดอร์ raw)
    try:
        from shutil import copy
        if not os.path.exists("new_week_data/new_data_18_09_2568.csv"):
             copy("raw/data_log_simulation_17_09_2568.csv", "new_week_data/new_data_18_09_2568.csv")
    except FileNotFoundError:
        print("คำเตือน: ไม่พบไฟล์ 'raw/data_log_simulation_17_09_2568.csv' เพื่อใช้จำลองข้อมูลใหม่")
    
    # 3. เริ่มกระบวนการ Fine-Tune
    fine_tune_weekly_model()