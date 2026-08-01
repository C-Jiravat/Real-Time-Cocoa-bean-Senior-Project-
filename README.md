# Cocoa Bean AI Platform

เว็บแอปพลิเคชันสำหรับตรวจสอบคุณภาพเมล็ดโกโก้ด้วย AI โดยใช้ YOLO ตรวจจับเมล็ด แล้วส่งภาพ crop เข้า ConvNeXt ONNX เพื่อจำแนกสีและข้อบกพร่อง

> Repository นี้ไม่เก็บ model weights, datasets, credential, virtual environment หรือผลลัพธ์ที่สร้างขึ้นจากการรันระบบ

## Features

- Login สำหรับผู้ดูแลระบบ
- วิเคราะห์ภาพที่อัปโหลด หรือรับภาพจากกล้อง Notebook / USB camera
- แสดง Bounding Box พร้อม class และ confidence
- จำแนกสี: ม่วง, น้ำตาล
- จำแนกข้อบกพร่อง: ปกติ, งอก, สีเทาหินชนวน, ขึ้นรา
- ประเมินชั้นคุณภาพเมล็ดและสรุปผลเป็น CSV
- Benchmark ด้วยภาพ + YOLO label หรือ Dataset ZIP
- แสดง Precision, Recall, F1-score, mAP และ Confusion Matrix

## Technology

- Frontend: React + Vite
- Backend: FastAPI
- Detection: YOLO ONNX
- Classification: ConvNeXt ONNX Runtime
- Image processing: OpenCV

## Project structure

```text
frontend/                    React user interface
backend/src/cocoa_platform/  FastAPI, inference, benchmark, grading
backend/config/              Model registry
docs/                        Public project documentation
assets/                      Optional screenshots for this README
```

## Local setup

### 1. Backend

Create a Python environment and install dependencies:

```powershell
.\backend\.venv-win\Scripts\python.exe -m pip install -r backend\requirements.txt -r backend\requirements-ml.txt
```

Create `.env` from `.env.example` and configure the administrator credentials locally. Do not commit `.env`.

Start the API:

```powershell
.\backend\.venv-win\Scripts\python.exe -m uvicorn cocoa_platform.api.app:app --app-dir backend/src --reload --port 8000
```

### 2. Frontend

```powershell
Set-Location frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:5173`.

## Model weights

Weights are intentionally excluded from Git. Configure their paths in `backend/config/model_registry.json` and keep the following local structure:

```text
weight/
├─ weight_yolo/
│  └─ best.onnx
├─ weight_color/
│  ├─ Phase_d03_best.onnx
│  └─ Phase_d03_best.onnx.data
└─ weight_defect/
   ├─ Phase3_WD0.15_best.onnx
   └─ Phase3_WD0.15_best.onnx.data
```

The `.onnx.data` files are required for ONNX models exported with external data.

## Benchmark dataset format

For color-only or defect-only evaluation:

```text
dataset.zip
├─ images/
└─ labels/
```

For combined color and defect evaluation:

```text
dataset.zip
├─ images/
├─ labels/color/
└─ labels/defect/
```

Labels use YOLO format:

```text
class_id x_center y_center width height
```


## Security note

Do not commit `.env`, access tokens, passwords, datasets, or model files. The detailed operational handoff document is intentionally ignored by Git.
