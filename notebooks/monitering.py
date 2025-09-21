# import psutil
# import time
# import csv

# # log ลงไฟล์ CSV
# with open("ram_log.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Time", "Used_RAM_MB", "Available_RAM_MB", "Percent_RAM"])

#     while True:
#         mem = psutil.virtual_memory()
#         writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
#                          mem.used / (1024**2),   # MB
#                          mem.available / (1024**2),
#                          mem.percent])
#         f.flush()
#         time.sleep(1)  # เก็บข้อมูลทุก 1 วินาที

import time
import csv
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # GPU ตัวแรก

with open("gpu_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "GPU_Usage_Percent", "GPU_Memory_Used_MB", "GPU_Memory_Total_MB"])

    while True:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)

        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                         util.gpu,
                         mem_info.used / (1024**2),
                         mem_info.total / (1024**2)])
        f.flush()
        time.sleep(1)  # เก็บทุก 1 วินาที