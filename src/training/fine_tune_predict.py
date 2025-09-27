import torch
import torch.nn as nn
import joblib
from pathlib import Path
import pandas as pd
import time 
import shutil
import tempfile

# กรุณาแก้ไข path และ import ให้ตรงกับโครงสร้างโปรเจคของคุณ
from src.models.lstm_model import LSTMModel
from src.data.generator import data_generator
from src.data.format_duration_time import format_duration

SCALER_REFERANCE_FOLDER = r"D:\Project_end\mainproject_fix\main_project\config\predict_model"
SCALER_FINE_TUNE_FOLDER = r"D:\Project_end\mainproject_fix\main_project\config\fine_tune"
MODEL_FOLDER = r"D:\Project_end\mainproject_fix\main_project\experiments\model_ex01"
NEW_DATA_FOLDER = r"D:\Project_end\mainproject_fix\main_project\data\raw" 
MODEL_FINE_TUNE_FOLDER = r"D:\Project_end\mainproject_fix\main_project\experiments\fine_tune_model"
# Hyperparameters สำหรับ Fine-Tuning
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
   
def fine_tune_scale():
    print("---  Start Fine-Tuning ---")
    
    try:
        input_scaler_path = Path(SCALER_REFERANCE_FOLDER) / "scaler_input.pkl"
        output_scaler_path = Path(SCALER_REFERANCE_FOLDER) / "scaler_output.pkl"
        scaler_input = joblib.load(input_scaler_path)
        scaler_output = joblib.load(output_scaler_path)

        # ค้นหาไฟล์ใหม่
        new_files = list(Path(NEW_DATA_FOLDER).glob("*.csv"))
        if not new_files:
            print("⚠️ File data not found Error")
            return

        # partial_fit scaler ด้วยข้อมูลใหม่
        for file_path in new_files:
            with pd.read_csv(file_path, chunksize=10000) as reader:
                for chunk in reader:
                    scaler_input.partial_fit(chunk[['DATA_INPUT']].values)
                    scaler_output.partial_fit(chunk[['DATA_OUTPUT']].values)

        # ตั้งชื่อไฟล์ zip ตามวันเวลา
        timestamp = time.strftime("%d_%m_%Y")
        zip_filename = Path(SCALER_FINE_TUNE_FOLDER) / f"scalers_{timestamp}.zip"

        # ใช้ temp dir เก็บไฟล์ pkl ก่อน zip
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_scaler_file = tmpdir_path / f"scaler_input_{timestamp}.pkl"
            output_scaler_file = tmpdir_path / f"scaler_output_{timestamp}.pkl"

            joblib.dump(scaler_input, input_scaler_file)
            joblib.dump(scaler_output, output_scaler_file)

            # zip ทั้ง folder ชั่วคราว
            shutil.make_archive(str(zip_filename.with_suffix('')), 'zip', tmpdir)

        print(f"✅ Fine-tuned scalers saved to: {zip_filename}")

    except FileNotFoundError:
        print(f"❌ File  config scaler not found Error")
        return
    

def fine_tune_model():
    try:
        # 🟢 โหลดจาก zip
        zip_files = list(Path(SCALER_FINE_TUNE_FOLDER).glob("*.zip"))
        if not zip_files:
            print("⚠️ ไม่พบไฟล์ zip ของ scaler")
            return

        latest_zip = max(zip_files, key=lambda p: p.stat().st_mtime)

        with tempfile.TemporaryDirectory() as tmpdir:
            shutil.unpack_archive(latest_zip, tmpdir, 'zip')
            tmpdir_path = Path(tmpdir)
            scaler_input = joblib.load(sorted(tmpdir_path.glob("scaler_input_*.pkl"))[0])
            scaler_output = joblib.load(sorted(tmpdir_path.glob("scaler_output_*.pkl"))[0])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ Loaded scalers from zip and ready for fine-tuning")
        model = LSTMModel(input_dim=2,
                      hidden_dim=HIDDEN_SIZE,
                      layer_dim=NUM_LAYERS,
                      output_dim=1).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_time = time.time()
        model.train()
        csv_files = list(Path(NEW_DATA_FOLDER).glob("*.csv"))
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
        timestamp = time.strftime("%d_%m_%Y")
        model_filename = Path(MODEL_FINE_TUNE_FOLDER) / f"lstm_model_{timestamp}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"📂 โมเดลถูกบันทึกที่ {model_filename}")


    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์โมเดลเดิม! กรุณารันสคริปต์ 02_train_model.py ก่อน")
        return


if __name__ == "__main__":
    fine_tune_scale()
    fine_tune_model()
