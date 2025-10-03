import torch
import torch.nn as nn
import joblib
from pathlib import Path
import time
import matplotlib.pyplot as plt

# -----------------------------
# Import โมเดลและ utilities
# -----------------------------
from src.models.lstm_model import LSTMModel
from src.data.generator import data_generator
from src.data.format_duration_time import format_duration


# -----------------------------
# Base Project Path (relative)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent  

SCALER_FOLDER = BASE_DIR / "config" / "predict_model"
DATA_FOLDER = BASE_DIR / "data" / "raw"
PATH_SAVE_MODEL = BASE_DIR / "experiments" / "model_ex01"

# -----------------------------
# Configurations
# -----------------------------
WINDOW_SIZE = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

PATIENCE =  20  # ถ้า loss ไม่ดีขึ้นเกิน 10 epoch → หยุด


# -----------------------------
# Training Utilities
# -----------------------------
def train_one_epoch(model, csv_files, criterion, optimizer,
                    scaler_input, scaler_output, device, window_size, batch_size):
    """ฝึกโมเดล 1 epoch"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    train_gen = data_generator(csv_files, scaler_input, scaler_output,
                               window_size, batch_size)

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
    return avg_loss


def save_checkpoint(model, path):
    """บันทึกโมเดล (state_dict เท่านั้น)"""
    torch.save(model.state_dict(), path)
    print(f"💾 โมเดลที่ดีที่สุดถูกบันทึกที่ {path}")


# -----------------------------
# Main Training Function
# -----------------------------
def train_model():
    print("--- 🚀 เริ่มการฝึกสอนโมเดล ---")

    # โหลด Scaler
    print("  -> โหลด Scaler ...")
    scaler_input = joblib.load(SCALER_FOLDER / "scaler_input.pkl")
    scaler_output = joblib.load(SCALER_FOLDER / "scaler_output.pkl")
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
    csv_files = list(DATA_FOLDER.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV ใน {DATA_FOLDER}")

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0
    loss_history = []

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        avg_loss = train_one_epoch(model, csv_files, criterion, optimizer,
                                   scaler_input, scaler_output, device,
                                   WINDOW_SIZE, BATCH_SIZE)

        loss_history.append(avg_loss)
        formatted_time = format_duration(time.time() - start_time)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"- Loss: {avg_loss:.6f} "
              f"- Time: {formatted_time}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # reset patience
            save_checkpoint(model, PATH_SAVE_MODEL / "lstm_model_best.pth")
        else:
            patience_counter += 1
            print(f"⏳ EarlyStopping patience: {patience_counter}/{PATIENCE}")

        # Early stopping condition
        if patience_counter >= PATIENCE:
            print("🛑 หยุดการฝึกเพราะ Loss ไม่ดีขึ้นตามที่กำหนด")
            break

    # -----------------------------
    # Plot Training Loss
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o', label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_plot = PATH_SAVE_MODEL / "Logging_image/training_loss.png"
    plt.savefig(save_plot)
    plt.close()
    print(f"📊 กราฟ Loss ถูกบันทึกที่ {save_plot}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    train_model()
