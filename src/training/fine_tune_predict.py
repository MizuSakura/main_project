import torch
import torch.nn as nn
import joblib
from pathlib import Path
import pandas as pd
import time
import shutil
import tempfile
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# --- Project Paths (relative) ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # ขึ้น 3 ระดับจากไฟล์ fine_tune.py
SCALER_REFERANCE_FOLDER = BASE_DIR / "config/predict_model"
SCALER_FINE_TUNE_FOLDER = BASE_DIR / "config/fine_tune"
MODEL_FOLDER = BASE_DIR / "experiments/model_ex01"
MODEL_FINE_TUNE_FOLDER = BASE_DIR / "experiments/fine_tune_model"
NEW_DATA_FOLDER = BASE_DIR / "data/raw"

# --- Hyperparameters ---
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# --- Imports from project ---
from src.models.lstm_model import LSTMModel
from src.data.generator import data_generator
from src.data.format_duration_time import format_duration


def fine_tune_scale():
    print("---  Start Fine-Tuning Scalers ---")
    try:
        input_scaler_path = SCALER_REFERANCE_FOLDER / "scaler_input.pkl"
        output_scaler_path = SCALER_REFERANCE_FOLDER / "scaler_output.pkl"
        scaler_input = joblib.load(input_scaler_path)
        scaler_output = joblib.load(output_scaler_path)

        # ค้นหาไฟล์ CSV ใหม่
        new_files = list(NEW_DATA_FOLDER.glob("*.csv"))
        if not new_files:
            print("⚠️ File data not found Error")
            return

        # partial_fit scaler ด้วยข้อมูลใหม่
        for file_path in new_files:
            for chunk in pd.read_csv(file_path, chunksize=10000):
                if 'DATA_INPUT' in chunk and 'DATA_OUTPUT' in chunk:
                    scaler_input.partial_fit(chunk[['DATA_INPUT']].values)
                    scaler_output.partial_fit(chunk[['DATA_OUTPUT']].values)
                else:
                    print(f"⚠️ Column missing in {file_path.name}")

        # ตั้งชื่อไฟล์ zip ตามวันเวลา
        timestamp = time.strftime("%d_%m_%Y_%H%M%S")
        zip_filename = SCALER_FINE_TUNE_FOLDER / f"scalers_{timestamp}.zip"

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

    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"❌ Error during fine-tune scalers: {e}")


def fine_tune_model():
    print("---  Start Fine-Tuning Model ---")
    try:
        # โหลด scalers จาก zip ล่าสุด
        zip_files = list(SCALER_FINE_TUNE_FOLDER.glob("*.zip"))
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

        model = LSTMModel(
            input_dim=2,
            hidden_dim=HIDDEN_SIZE,
            layer_dim=NUM_LAYERS,
            output_dim=1
        ).to(device)

        # โหลด pretrained model ถ้ามี
        model_files = list(MODEL_FOLDER.glob("*.pth"))
        if not model_files:
            print("⚠️ ไม่พบ pretrained model ใน MODEL_FOLDER, จะฝึกจาก scratch")
        else:
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            model.load_state_dict(torch.load(latest_model, map_location=device))
            print(f"✅ Loaded pretrained model: {latest_model}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        writer = SummaryWriter(log_dir=BASE_DIR/'runs/fine_tune')

        csv_files = list(NEW_DATA_FOLDER.glob("*.csv"))
        if not csv_files:
            print("⚠️ ไม่พบไฟล์ CSV สำหรับ train model")
            return

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            batch_count = 0

            train_gen = data_generator(csv_files, scaler_input, scaler_output,
                                       WINDOW_SIZE, BATCH_SIZE)

            for X_batch, y_batch in tqdm(train_gen, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                X_batch = X_batch.float().to(device)
                y_batch = y_batch.float().to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / max(1, batch_count)
            writer.add_scalar('Loss/train', avg_loss, epoch)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.6f}")

        total_time = time.time() - start_time
        formatted_time = format_duration(total_time)
        print(f"✅ Training finished. Total time: {formatted_time}")

        # บันทึกโมเดล fine-tune ใหม่
        timestamp = time.strftime("%d_%m_%Y_%H%M%S")
        model_filename = MODEL_FINE_TUNE_FOLDER / f"lstm_model_{timestamp}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"📂 Fine-tuned model saved at {model_filename}")

        writer.close()

    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"❌ Error during fine-tune model: {e}")
        return


if __name__ == "__main__":
    fine_tune_scale()
    fine_tune_model()
