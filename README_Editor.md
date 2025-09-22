# เอกสารประกอบการใช้งาน Git

เอกสารนี้สรุปแนวทางปฏิบัติที่ดีที่สุดสำหรับการใช้งาน Git ในโปรเจกต์ เพื่อให้การทำงานร่วมกันเป็นไปอย่างราบรื่นและสามารถตรวจสอบย้อนหลังได้ง่าย (traceability)

---

## ✍️ แนวทางการเขียน Commit Message

การเขียน Commit Message ที่ดีมีความสำคัญอย่างยิ่งต่อการทำงานร่วมกัน ทำให้เพื่อนร่วมทีมเข้าใจการเปลี่ยนแปลงในโค้ดได้อย่างรวดเร็ว โดยไม่ต้องไล่อ่านโค้ดทั้งหมด เราจะใช้ **Conventional Commits** เป็นมาตรฐาน

### โครงสร้างของ Commit Message

1.  **Subject (หัวเรื่อง)**: เป็นส่วนที่สำคัญที่สุด ต้องสั้น กระชับ และสื่อความหมาย
    * **`<type>`**: ชนิดของการ commit (ดูรายการด้านล่าง)
    * **`<scope>`** (ถ้ามี): ส่วนของโปรเจกต์ที่เกี่ยวข้อง เช่น `auth`, `db`, `ui`
    * **`<description>`**: คำอธิบายการเปลี่ยนแปลงแบบสรุป (ขึ้นต้นด้วยตัวพิมพ์เล็กและไม่ต้องมีจุดจบประโยค)

2.  **Body (เนื้อหา)** (ถ้ามี): ใช้อธิบายรายละเอียดเพิ่มเติม ว่า "ทำไม" ถึงเปลี่ยนแปลง และแนวทางที่ใช้คืออะไร

3.  **Footer (ส่วนท้าย)** (ถ้ามี): ใช้อ้างอิงถึง Issue Tracker เช่น `Closes #123` หรือระบุ Breaking Changes

---

### ประเภทของ Commit (Type)

* **`feat`**: เพิ่มฟีเจอร์ใหม่ (a new feature)
* **`fix`**: แก้ไขข้อผิดพลาด (a bug fix)
* **`docs`**: แก้ไขเอกสารประกอบโปรเจกต์เท่านั้น (documentation only changes)
* **`style`**: แก้ไขการจัดรูปแบบโค้ดที่ไม่กระทบต่อ logic (e.g., white-space, formatting, semi-colons)
* **`refactor`**: ปรับแก้โค้ดที่ไม่ใช่การเพิ่มฟีเจอร์หรือแก้บั๊ก (code refactoring)
* **`test`**: เพิ่มหรือแก้ไข test case
* **`chore`**: การเปลี่ยนแปลงอื่นๆ ที่ไม่เกี่ยวกับ source code หรือ test เช่น การแก้ไข build script, จัดการ dependency

---
### ตัวอย่าง Commit Message ที่ดี

**ตัวอย่าง 1: การเพิ่มฟีเจอร์**
ได้เลยครับ นี่คือเนื้อหาทั้งหมดในรูปแบบไฟล์ Markdown ที่คุณสามารถคัดลอกไปวางในไฟล์ README.md ของคุณได้ทันทีครับ

Markdown

# เอกสารประกอบการใช้งาน Git

เอกสารนี้สรุปแนวทางปฏิบัติที่ดีที่สุดสำหรับการใช้งาน Git ในโปรเจกต์ เพื่อให้การทำงานร่วมกันเป็นไปอย่างราบรื่นและสามารถตรวจสอบย้อนหลังได้ง่าย (traceability)

---

## ✍️ แนวทางการเขียน Commit Message

การเขียน Commit Message ที่ดีมีความสำคัญอย่างยิ่งต่อการทำงานร่วมกัน ทำให้เพื่อนร่วมทีมเข้าใจการเปลี่ยนแปลงในโค้ดได้อย่างรวดเร็ว โดยไม่ต้องไล่อ่านโค้ดทั้งหมด เราจะใช้ **Conventional Commits** เป็นมาตรฐาน

### โครงสร้างของ Commit Message

<type>[optional scope]: <description>

[optional body]

[optional footer]


1.  **Subject (หัวเรื่อง)**: เป็นส่วนที่สำคัญที่สุด ต้องสั้น กระชับ และสื่อความหมาย
    * **`<type>`**: ชนิดของการ commit (ดูรายการด้านล่าง)
    * **`<scope>`** (ถ้ามี): ส่วนของโปรเจกต์ที่เกี่ยวข้อง เช่น `auth`, `db`, `ui`
    * **`<description>`**: คำอธิบายการเปลี่ยนแปลงแบบสรุป (ขึ้นต้นด้วยตัวพิมพ์เล็กและไม่ต้องมีจุดจบประโยค)

2.  **Body (เนื้อหา)** (ถ้ามี): ใช้อธิบายรายละเอียดเพิ่มเติม ว่า "ทำไม" ถึงเปลี่ยนแปลง และแนวทางที่ใช้คืออะไร

3.  **Footer (ส่วนท้าย)** (ถ้ามี): ใช้อ้างอิงถึง Issue Tracker เช่น `Closes #123` หรือระบุ Breaking Changes

---

### ประเภทของ Commit (Type)

* **`feat`**: เพิ่มฟีเจอร์ใหม่ (a new feature)
* **`fix`**: แก้ไขข้อผิดพลาด (a bug fix)
* **`docs`**: แก้ไขเอกสารประกอบโปรเจกต์เท่านั้น (documentation only changes)
* **`style`**: แก้ไขการจัดรูปแบบโค้ดที่ไม่กระทบต่อ logic (e.g., white-space, formatting, semi-colons)
* **`refactor`**: ปรับแก้โค้ดที่ไม่ใช่การเพิ่มฟีเจอร์หรือแก้บั๊ก (code refactoring)
* **`test`**: เพิ่มหรือแก้ไข test case
* **`chore`**: การเปลี่ยนแปลงอื่นๆ ที่ไม่เกี่ยวกับ source code หรือ test เช่น การแก้ไข build script, จัดการ dependency

---

### ตัวอย่าง Commit Message ที่ดี

**ตัวอย่าง 1: การเพิ่มฟีเจอร์**

feat(auth): add google login functionality

Implement OAuth 2.0 for Google authentication.
This allows users to sign in with their Google account.

Closes #42


**ตัวอย่าง 2: การแก้ไขบั๊ก**

fix(api): correct calculation for user score

The previous calculation was off by a factor of 2.
This commit corrects the formula and adds a unit test to prevent regressions.


**ตัวอย่าง 3: การแก้ไขเอกสาร**

docs: update README with git commit guidelines


---

## ⚙️ คำสั่ง Git ที่จำเป็น

### การตั้งค่าเริ่มต้น (Initial Setup)

* **`git clone [url]`**: คัดลอก repository จาก remote server มายังเครื่อง local
    ```bash
    git clone [https://github.com/username/repository.git](https://github.com/username/repository.git)
    ```
* **`git init`**: สร้าง repository ใหม่ในโฟลเดอร์ปัจจุบัน

---

### การทำงานพื้นฐาน (Basic Workflow)

* **`git status`**: ตรวจสอบสถานะของไฟล์ใน working directory และ staging area
* **`git add [file]`**: เพิ่มไฟล์ที่ต้องการเข้าไปใน staging area เพื่อเตรียม commit
    ```bash
    # เพิ่มไฟล์เดียว
    git add src/main.py
    
    # เพิ่มทุกไฟล์ที่มีการเปลี่ยนแปลง
    git add .
    ```
* **`git commit -m "Your message"`**: บันทึกการเปลี่ยนแปลง (snapshot) จาก staging area ไปยัง local repository
    ```bash
    git commit -m "feat: add user profile page"
    ```
* **`git push`**: ส่งข้อมูล commit จาก local repository ไปยัง remote repository
* **`git pull`**: ดึงข้อมูลล่าสุดจาก remote repository มายัง local repository

---

### การย้อนกลับและแก้ไข (Undoing Changes)

* **`git reset [file]`**: นำไฟล์ออกจาก staging area แต่ยังคงการเปลี่ยนแปลงไว้ใน working directory
    ```bash
    # Unstage a file
    git reset HEAD~
    ```
* **`git reset --hard [commit]`**: **(⚠️ ระวัง: คำสั่งนี้ลบข้อมูลอย่างถาวร)** ย้อนกลับไปยัง commit ที่ระบุและลบการเปลี่ยนแปลงทั้งหมดหลังจากนั้นทิ้งไป
    ```bash
    # กลับไปที่ commit ก่อนหน้า 1 commit
    git reset --hard HEAD~1
    ```
* **`git revert [commit]`**: สร้าง commit ใหม่เพื่อยกเลิกการเปลี่ยนแปลงของ commit ที่ระบุ เป็นวิธีที่ปลอดภัยกว่า `git reset` สำหรับ public history

---

### การตรวจสอบประวัติ (Inspecting History)

* **`git log`**: แสดงประวัติการ commit ทั้งหมด
    ```bash
    # ดู log แบบสั้นๆ บรรทัดเดียว
    git log --oneline
    
    # ดู log พร้อมกราฟของ branch
    git log --graph --oneline --all
    ```
* **`git diff`**: แสดงความแตกต่างระหว่าง working directory และ staging area
* **`git diff --staged`**: แสดงความแตกต่างระหว่าง staging area และ commit ล่าสุด