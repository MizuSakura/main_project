# เอกสารประกอบโปรเจกต์

โครงสร้างของ repository นี้ถูกออกแบบตามแนวทางปฏิบัติที่ดีสำหรับการทำงานวิจัยเชิงข้อมูล (data-driven research) และการพัฒนาระบบ Machine Learning  
มีวัตถุประสงค์เพื่อให้การทำงานมี **ความสามารถในการทำซ้ำ (reproducibility)**, **การขยายระบบ (scalability)** และ **การดูแลรักษา (maintainability)**

---

## โครงสร้างโฟลเดอร์

### `data/`
เก็บข้อมูลทั้งหมดที่เกี่ยวข้องกับโปรเจกต์ แบ่งออกเป็น 3 ส่วนย่อย:
- **`raw/`** : ข้อมูลดิบจากแหล่งต้นทาง (เช่น sensor logs, CSV, ไฟล์ภาพ) โดย **ห้ามแก้ไข** เพื่อรักษาความถูกต้องของข้อมูลต้นฉบับ  
- **`processed/`** : ข้อมูลที่ผ่านการประมวลผลแล้ว เช่น cleaned, normalized, หรือ transformed data สำหรับใช้ในการ train/test  
- **`manifests/`** : เก็บ metadata และ schema ของข้อมูล เช่น data dictionary, ไฟล์อธิบายโครงสร้างข้อมูล เพื่อรองรับการทำซ้ำงานวิจัย  

---

### `experiments/`
เก็บผลลัพธ์การทดลอง ได้แก่:
- checkpoints ของโมเดล  
- training/evaluation logs  
- ค่า metrics และผลการประเมิน  
- ไฟล์ config ของการทดลอง  

**คำแนะนำ:** ให้สร้างโฟลเดอร์ย่อยสำหรับแต่ละการทดลอง เช่น `exp_01/`, `exp_02/` เพื่อความเป็นระบบและตรวจสอบย้อนหลังได้ง่าย

---

### `notebooks/`
เก็บ Jupyter notebooks สำหรับ:
- การวิเคราะห์ข้อมูลเบื้องต้น (Exploratory Data Analysis – EDA)  
- การสร้าง visualization  
- การสร้างและทดสอบโมเดลเบื้องต้น  
- ใช้เป็นสมุดบันทึกการทำวิจัย (research log)  

---

### `scripts/`
เก็บสคริปต์ที่ทำงานอัตโนมัติหรือเป็นงานเฉพาะ เช่น:
- preprocessing data  
- training script  
- evaluation pipeline  
- utility script  

สคริปต์เหล่านี้ควรมีหน้าที่ชัดเจนและใช้งานได้โดยตรง

---

### `SETUP_PROJECT/`
เก็บไฟล์สำหรับการตั้งค่า environment ของโปรเจกต์ เช่น:
- ไฟล์ dependency ที่เกี่ยวข้อง  
- ตัวช่วยจัดการ environment  
- ไฟล์หรือ script ที่เกี่ยวข้องกับการ setup เบื้องต้น  

---

### `src/`
เก็บ **source code หลัก** ของโปรเจกต์ โดยแบ่งเป็นโมดูล เช่น:
- data loader และ preprocessing pipeline  
- model definitions  
- training/optimization logic  
- utility functions  

โฟลเดอร์นี้เป็นหัวใจหลักของระบบ

---

### `tests/`
เก็บโค้ดสำหรับทดสอบ เพื่อยืนยันความถูกต้องและความเสถียรของโค้ดใน `src/`  
- Unit test (ทดสอบฟังก์ชันย่อย)  
- Integration test (ทดสอบการทำงานรวม)  
- ควรใช้ framework เช่น `pytest` หรือ `unittest`  

---

## ไฟล์ที่ระดับ Root

- **`.gitignore`** : ระบุไฟล์หรือโฟลเดอร์ที่ไม่ควรถูก track โดย Git (เช่น cache, log, dataset ขนาดใหญ่)  
- **`requirements.txt`** : ไฟล์แสดง dependencies ที่โปรเจกต์ต้องใช้  
  > ติดตั้งได้โดยใช้คำสั่ง:  
  ```bash
  pip install -r requirements.txt
  pip freeze > requirements.txt