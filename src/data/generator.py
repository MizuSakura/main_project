import pandas as pd
import numpy as np
import torch
from src.data import create_sequences


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