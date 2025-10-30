# Face Attendance (FastAPI)

Features:
- Enroll students (face + metadata)
- Real-time attendance from webcam
- Logs per class/date/time; admin UI + CSV export
- Basic anti-spoofing (mediapipe heuristics), logging, privacy safeguards

## Quickstart
1. Create venv (Windows PowerShell):
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
2. Install deps:
   - pip install -r requirements.txt
3. Run server:
   - uvicorn app.main:app --reload --port 8000
4. Open UI:
   - http://localhost:8000

## Notes
- Embeddings via InsightFace (ONNXRuntime CPU). First run downloads models.
- DB: SQLite at `data/attendance.db`.
- Privacy: consent checkbox at enrollment. See `app/config.py` for thresholds.
- Export CSV from Admin > Logs.
