import sys
import subprocess
from fastapi import FastAPI, HTTPException
from pathlib import Path
import os

PYTHON_EXECUTABLE = sys.executable 
# BASE_DIR = Path(__file__).parent
# SCRIPT_WORKING_DIR = BASE_DIR / "ex01_peperation_data"
# RESCALE_SCRIPT_PATH = SCRIPT_WORKING_DIR / "Rescal_data.py"
# TRAIN_SCRIPT_PATH = SCRIPT_WORKING_DIR / "Model_LSTM.py"

RESCALE_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Rescal_data.py")
TRAIN_SCRIPT_PATH = Path(r"C:\Users\kitta\OneDrive\Desktop\INIT\main_project\notebooks\ex01_peperation_data\Model_LSTM.py")
SCRIPT_WORKING_DIR = RESCALE_SCRIPT_PATH.parent

app = FastAPI(title="AI Factory Controller")

def run_script_and_capture(script_path: Path):
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"FATAL: Hardcoded script path does not exist! Check your path: {script_path}")
    
    try:
        # สร้าง environment variables ใหม่, สืบทอดของเก่ามาด้วย
        script_env = os.environ.copy()
        # บังคับให้ Python interpreter ที่ถูกเรียกใช้ UTF-8 เท่านั้น!
        script_env["PYTHONUTF8"] = "1"

        process = subprocess.run(
            [PYTHON_EXECUTABLE, "-u", str(script_path.name)],
            cwd=str(SCRIPT_WORKING_DIR),
            capture_output=True,
            text=True,
            # encoding='utf-8', # <--- เราไม่ต้องใช้ encoding ตรงนี้แล้ว
            check=False,
            env=script_env # <--- เพิ่มบรรทัดนี้เข้าไป!
        )
        
        full_log = f"--- SCRIPT OUTPUT ---\n{process.stdout}\n\n--- SCRIPT ERROR ---\n{process.stderr}"
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=full_log)
        
        return {"message": "Script finished successfully!", "log": full_log}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastAPI Internal Error: {str(e)}")


@app.post("/start-scaling")
async def trigger_scaling():
    return run_script_and_capture(RESCALE_SCRIPT_PATH)

@app.post("/start-training")
async def trigger_training():
    return run_script_and_capture(TRAIN_SCRIPT_PATH)