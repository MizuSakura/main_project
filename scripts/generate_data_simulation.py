import numpy as np
import matplotlib.pyplot as plt
from Logging_andplot import Logger
import numpy as np
import time

class RC_Tank_Env:
    """
    จำลองถังน้ำที่สามารถเลือกโหมดการควบคุมได้
    เพิ่มการจำกัดค่า Action สูงสุดสำหรับแต่ละโหมด
    """
    def __init__(self, R=1.5, C=2.0, dt=0.1, 
                 control_mode='current',
                 setpoint_level=5.0,
                 level_max=10.0,
                 max_action_volt=24.0,   # << เพิ่ม: ขีดจำกัด Action สำหรับโหมด voltage
                 max_action_current=5.0): # << เพิ่ม: ขีดจำกัด Action สำหรับโหมด current
        
        if control_mode not in ['voltage', 'current']:
            raise ValueError("control_mode must be either 'voltage' or 'current'")
            
        self.mode = control_mode
        self.R = R
        self.C = C
        self.dt = dt
        self.level_max = level_max
        self.setpoint_level = setpoint_level
        
        # << เพิ่ม: เก็บค่าขีดจำกัดของ Action
        self.max_action_volt = max_action_volt
        self.max_action_current = max_action_current

        self.level = 0.0
        self.time = 0.0
        self.reset()

    def reset(self,defualt=None):
        self.level = np.random.uniform(low=0, high=self.level_max)
        if defualt is not None:
            self.level = defualt
        self.time = 0.0
        done = False
        return float(self.level), done
    
    def step(self, action):
        # << Clip: จำกัดค่า Action ที่รับเข้ามาตามโหมดที่เลือก
        # โดยสมมติว่าค่า Action เป็นบวกเสมอ (0 ถึง max)
        if self.mode == 'voltage':
            action = np.clip(action, 0, self.max_action_volt)
        elif self.mode == 'current':
            action = np.clip(action, 0, self.max_action_current)

        delta_level = 0.0
        if self.mode == 'voltage':
            # Action คือ "voltage_source"
            voltage_source = action
            current = (voltage_source - self.level) / self.R
            delta_level = (current / self.C) * self.dt
            
        elif self.mode == 'current':
            # Action คือ "inflow_rate"
            inflow_rate = action
            outflow_rate = self.level / self.R
            net_flow = inflow_rate - outflow_rate
            delta_level = (net_flow / self.C) * self.dt

        self.level += delta_level
        self.level = np.clip(self.level, 0, self.level_max)
        self.time += self.dt

        error = self.setpoint_level - self.level
        done = abs(error) <= 0.1

        return float(self.level), done
    
class SignalGenerator:
    def __init__(self, t_end=10, dt=0.01):
        self.t = np.arange(0, t_end, dt)

    def step(self, amplitude=1, start_time=0):
        return (self.t >= start_time) * amplitude

    def impulse(self, amplitude=1, at_time=0):
        signal = np.zeros_like(self.t)
        idx = np.argmin(np.abs(self.t - at_time))
        signal[idx] = amplitude / (self.t[1] - self.t[0])  # scaling ให้ area ≈ amplitude
        return signal

    def pulse(self, amplitude=1, start_time=0, width=1):
        return ((self.t >= start_time) & (self.t <= start_time+width)) * amplitude
    
    def pwm(self, amplitude=1, freq=1, duty=0.5):
        T = 1 / freq
        return amplitude * ((self.t % T) < duty * T)

    def ramp(self, slope=1, start_time=0):
        return slope * np.maximum(0, self.t - start_time)

    def parabolic(self, coeff=1, start_time=0):
        return coeff * np.maximum(0, self.t - start_time)**2
if __name__ == '__main__':
    TIME_SIMULATION = 1000        # ลดเวลาจำลองให้เหมาะสม
    VOLT_SUPPLY = 24
    R = 2000 #Ohm
    C = 0.1  #F
    DT = 0.01
    SETPOINT = 24
    MAX_CURRENT = 3 #A
    MODE_CONTROL = 'voltage' # หรือ 'current'
    FOLDER = r"D:\Project_end\mainproject_fix\main_project\data\raw"
    FOLDER_PICTURE = r"D:\Project_end\mainproject_fix\main_project\data\picture"
    PWM = np.arange(0,1.1,0.1)
    File_name = np.arange(20,30,1)

    LOG_DT = 0.1   
    log_interval = int(LOG_DT / DT)  # กี่ simulation step ต่อการ log 1 ครั้ง

    for i in range(0,10):
        FILE_NAME = str(File_name[i])
        sg = SignalGenerator(t_end=TIME_SIMULATION, dt=DT)
        env = RC_Tank_Env(R=R, C=C, dt=DT, control_mode=MODE_CONTROL,
                          setpoint_level=SETPOINT,
                          max_action_volt=VOLT_SUPPLY,
                          max_action_current=MAX_CURRENT,
                          level_max=VOLT_SUPPLY)
        logger = Logger()
        env.reset(defualt=0)

        if MODE_CONTROL == 'voltage':
            soure = VOLT_SUPPLY
        else:
            soure = MAX_CURRENT

        DATA_INPUT = sg.pwm(amplitude=1, freq=0.001, duty=PWM[i])
        DATA_OUTPUT, ACTION, TIME_LOG = [], [], []

        for idx, signal in enumerate(DATA_INPUT):
            output = soure * signal
            v_out, done = env.step(action=output)

            # << log ตาม sampling rate ที่กำหนด
            if idx % log_interval == 0:
                ACTION.append(output)
                DATA_OUTPUT.append(v_out)
                TIME_LOG.append(idx * DT)

        logger.add_data_log(
            columns_name=["TIME", "DATA_INPUT", "DATA_OUTPUT"],
            data_list=[TIME_LOG, ACTION, DATA_OUTPUT])
        logger.save_to_csv(file_name=FILE_NAME, folder_name=FOLDER)

        # ใช้ TIME_LOG แทน sg.t เพื่อพล็อต
        plt.plot(TIME_LOG, DATA_OUTPUT, label="DATA_OUTPUT")
        plt.plot(TIME_LOG, ACTION, label="DATA_INPUT SCALE", alpha=0.5)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Test Input Signals (duty={PWM[i]:.1f})")
        plt.grid(True)
        plt.show()
        plt.savefig(f"{FOLDER}/{FILE_NAME}.png")
        time.sleep(1)
        plt.close()
