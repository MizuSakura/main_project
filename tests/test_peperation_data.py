import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import os
from pathlib import Path

# =============================================================================
# Section 1: The "Gold Standard" Scaler Fitting
# (ทำขั้นตอนนี้แค่ครั้งเดียว หรือเมื่อมีไฟล์ข้อมูลใหม่เพิ่มเข้ามา)
# =============================================================================
def create_global_scaler(folder_path, chunk_size=10000):
    """
    วนอ่านไฟล์ CSV ทั้งหมดในโฟลเดอร์ทีละส่วน (chunk) เพื่อ fit scaler
    โดยไม่ใช้ RAM เยอะด้วยเทคนิค partial_fit
    
    Args:
        folder_path (str): ตำแหน่งโฟลเดอร์ที่เก็บไฟล์ CSV
        chunk_size (int): จำนวนแถวที่จะอ่านในแต่ละครั้ง
        
    Returns:
        tuple: (scaler_input, scaler_output) ที่ fit เสร็จแล้ว
    """
    print("--- 🚀 Starting Global Scaler Fitting Process ---")
    
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()
    
    # ค้นหาไฟล์ .csv ทั้งหมดในโฟลเดอร์
    csv_files = list(Path(folder_path).glob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV files found in '{folder_path}'")
        return None, None
        
    print(f"Found {len(csv_files)} files to process.")

    # วนลูปไปทีละไฟล์
    for i, file_path in enumerate(csv_files):
        print(f"  -> Processing file {i+1}/{len(csv_files)}: {file_path.name}")
        
        # ใช้ pd.read_csv แบบ iterator เพื่ออ่านทีละ chunk
        with pd.read_csv(file_path, chunksize=chunk_size) as reader:
            for chunk in reader:
                # ดึงข้อมูลเฉพาะคอลัมน์ที่สนใจ
                input_chunk = chunk[['DATA_INPUT']].values
                output_chunk = chunk[['DATA_OUTPUT']].values
                
                # *** หัวใจของเทคนิคนี้ ***
                scaler_input.partial_fit(input_chunk)
                scaler_output.partial_fit(output_chunk)

    print("\n--- ✅ Global Scaler Fitting Complete! ---")
    print(f"Input Scaler Range: {scaler_input.data_min_[0]:.2f} to {scaler_input.data_max_[0]:.2f}")
    print(f"Output Scaler Range: {scaler_output.data_min_[0]:.2f} to {scaler_output.data_max_[0]:.2f}")
    
    return scaler_input, scaler_output

# --- สมมติว่าไฟล์ของคุณอยู่ในโฟลเดอร์ชื่อ 'raw' ---
DATA_FOLDER = "raw" 
# สร้างโฟลเดอร์และไฟล์จำลอง (หากยังไม่มี)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    # สร้างไฟล์จำลอง 3 ไฟล์
    for i in range(3):
        n_points = 2000
        input_data = np.random.rand(n_points, 1) * (i + 1) * 10
        output_data = np.sin(np.arange(n_points) / 50) * (i + 1) * 20 + 30
        df = pd.DataFrame({'DATA_INPUT': input_data.flatten(), 'DATA_OUTPUT': output_data.flatten()})
        df.to_csv(f"{DATA_FOLDER}/data_log_sim_{i+1}.csv", index=False)

# **รันฟังก์ชันนี้เพื่อสร้าง Scaler ของเรา**
global_scaler_input, global_scaler_output = create_global_scaler(DATA_FOLDER)


# =============================================================================
# Section 2: Using the Pre-fitted Scaler in a Memory-Efficient Generator
# (นำ Scaler ที่ได้จาก Section 1 มาใช้ในตอนเทรนโมเดล)
# =============================================================================
def data_generator_with_global_scaler(file_list, scaler_in, scaler_out, window_size):
    """
    Generator ที่ใช้ Scaler ซึ่งถูก fit มาแล้วอย่างสมบูรณ์
    """
    for file_path in file_list:
        df = pd.read_csv(file_path)
        data = df[['DATA_INPUT', 'DATA_OUTPUT']].values.astype(np.float32)
        
        # *** ไม่มีการ .fit() อีกต่อไป ใช้ .transform() เท่านั้น ***
        scaled_input = scaler_in.transform(data[:, 0].reshape(-1, 1))
        scaled_output = scaler_out.transform(data[:, 1].reshape(-1, 1))
        scaled_data = np.hstack([scaled_input, scaled_output])
        
        X, y = create_sequences(scaled_data, window_size) # (สมมติว่ามีฟังก์ชันนี้อยู่)
        
        yield torch.from_numpy(X).float(), torch.from_numpy(y).float().view(-1, 1)

# --- ตอนนำไปใช้งาน ---
if global_scaler_input:
    file_paths = list(Path(DATA_FOLDER).glob("*.csv"))
    
    # สร้าง generator ที่พร้อมใช้งาน
    my_generator = data_generator_with_global_scaler(
        file_list=file_paths,
        scaler_in=global_scaler_input,
        scaler_out=global_scaler_output,
        window_size=30
    )
    
    # คุณสามารถนำ my_generator ไปใช้ใน training loop ได้เลย
    print("\nGenerator is ready to be used for training with the global scaler.")
    # for X_batch, y_batch in my_generator:
    #     # ... training logic ...
    #     pass