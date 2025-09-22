import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import joblib
import os
from pathlib import Path

DATA_FOLDER = r"D:\Project_end\mainproject\data\raw" 
FOLDER_PATH_SAVE = r"D:\Project_end\mainproject\experiments\config\predict_model"

CHUNK_SIZE = 10000

if not os.path.exists(DATA_FOLDER):
    print(f"❌ Error: Folder '{DATA_FOLDER}' not found. Please check the path.")

def find_scale_referance(folder_path,folder_path_save, chunk_size=10000):
    print("--- 🚀 Starting Global Scaler Fitting Process ---")

    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()

    csv_files = list(Path(folder_path).glob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV files found in '{folder_path}'")
        return None, None
    
    print(f"Found {len(csv_files)} files to process.")

    for i, file_path in enumerate(csv_files):
        print(f"  -> Processing file {i+1}/{len(csv_files)}: {file_path.name}")
        
        # ใช้ pd.read_csv แบบ iterator เพื่ออ่านทีละ chunk
        with pd.read_csv(file_path, chunksize=chunk_size) as reader:
            for chunk in reader:
                input_chunk = chunk[['DATA_INPUT']].values
                output_chunk = chunk[['DATA_OUTPUT']].values

                scaler_input.partial_fit(input_chunk)
                scaler_output.partial_fit(output_chunk)
    
    print("\n--- Scaler Fitting Complete! ---")
    print(f"Input Scaler Range: {scaler_input.data_min_[0]:.2f} to {scaler_input.data_max_[0]:.2f}")
    print(f"Output Scaler Range: {scaler_output.data_min_[0]:.2f} to {scaler_output.data_max_[0]:.2f}")

    input_scaler_path = Path(folder_path_save) / "scaler_input.pkl"
    output_scaler_path = Path(folder_path_save) / "scaler_output.pkl"
    
    joblib.dump(scaler_input, input_scaler_path)
    joblib.dump(scaler_output, output_scaler_path)
    print(f"\n save Scaler to '{folder_path_save}'")

    return scaler_input, scaler_output


if __name__ == '__main__':
    global_scaler_input, global_scaler_output = find_scale_referance(DATA_FOLDER,FOLDER_PATH_SAVE,CHUNK_SIZE)