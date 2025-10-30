import io
import json
import os
from datetime import date, datetime, time
from typing import List, Optional

import orjson
import numpy as np
from fastapi import FastAPI, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, SQLModel, select

from .config import settings
from .database import engine, init_db
from .models import Class, Student, Session as Sess, Attendance
from .face import image_bytes_to_bgr, embed_image_bgr, cosine_distance, save_face_thumb, mediapipe_liveness_heuristic

app = FastAPI(title="Face Attendance")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

init_db()

# Utility DB session
def get_db():
    with Session(engine) as session:
        yield session

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    classes = db.exec(select(Class)).all()
    return templates.TemplateResponse("home.html", {"request": request, "classes": classes})

@app.post("/admin/class/create")
def create_class(name: str = Form(...), code: str = Form(...), db: Session = Depends(get_db)):
    cl = Class(name=name, code=code)
    db.add(cl)
    db.commit()
    db.refresh(cl)
    return RedirectResponse(url="/", status_code=303)

@app.get("/enroll", response_class=HTMLResponse)
def enroll_page(request: Request, class_id: Optional[int] = None, db: Session = Depends(get_db)):
    classes = db.exec(select(Class)).all()
    return templates.TemplateResponse("enroll.html", {"request": request, "classes": classes, "class_id": class_id})

@app.post("/api/enroll")
def api_enroll(
    name: str = Form(...),
    student_id: str = Form(...),
    class_id: Optional[int] = Form(None),
    consent: Optional[bool] = Form(False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    img_bytes = file.file.read()
    img = image_bytes_to_bgr(img_bytes)
    emb_res = embed_image_bgr(img)
    if not emb_res:
        return JSONResponse({"ok": False, "error": "No face detected or low score."}, status_code=400)
    emb, det_score = emb_res
    live_ok = True
    if settings.liveness_required:
        live_ok = mediapipe_liveness_heuristic(img)
    if not live_ok:
        return JSONResponse({"ok": False, "error": "Liveness check failed."}, status_code=400)

    thumb_path = os.path.join(settings.face_thumb_dir, f"{student_id}_{int(datetime.utcnow().timestamp())}.jpg")
    save_face_thumb(img, thumb_path)

    st = Student(
        name=name,
        student_id=student_id,
        class_id=class_id,
        embedding_json=json.dumps(emb.tolist()),
        face_thumb_path=thumb_path,
    )
    db.add(st)
    db.commit()
    db.refresh(st)
    return JSONResponse({"ok": True, "student_id": st.id})

@app.get("/recognize", response_class=HTMLResponse)
def recognize_page(request: Request, class_id: Optional[int] = None, db: Session = Depends(get_db)):
    classes = db.exec(select(Class)).all()
    return templates.TemplateResponse("recognize.html", {"request": request, "classes": classes, "class_id": class_id})

@app.post("/api/recognize_frame")
def api_recognize_frame(
    class_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    img_bytes = file.file.read()
    img = image_bytes_to_bgr(img_bytes)
    emb_res = embed_image_bgr(img)
    if not emb_res:
        return JSONResponse({"ok": False, "recognized": []})
    emb, det_score = emb_res

    q = select(Student)
    if class_id:
        q = q.where(Student.class_id == class_id)
    students = db.exec(q).all()
    best = None
    best_dist = 1e9
    for s in students:
        ref = np.array(json.loads(s.embedding_json), dtype=np.float32)
        dist = cosine_distance(emb, ref)
        if dist < best_dist:
            best_dist = dist
            best = s

    recognized = []
    if best and best_dist <= settings.embedding_threshold:
        # Ensure session for today
        today = date.today()
        existing = db.exec(select(Sess).where(Sess.class_id == (best.class_id or 0), Sess.session_date == today)).first()
        if not existing:
            existing = Sess(class_id=(best.class_id or 0), session_date=today, start_time=time(hour=0, minute=0))
            db.add(existing)
            db.commit()
            db.refresh(existing)
        att = Attendance(session_id=existing.id, student_id=best.id, confidence=1 - best_dist, liveness_passed=True)
        db.add(att)
        db.commit()
        recognized.append({"student_id": best.student_id, "name": best.name, "dist": best_dist, "db_id": best.id})

    return JSONResponse({"ok": True, "recognized": recognized})

@app.get("/admin/logs", response_class=HTMLResponse)
def admin_logs(request: Request, class_id: Optional[int] = None, day: Optional[str] = None, db: Session = Depends(get_db)):
    classes = db.exec(select(Class)).all()
    q = select(Attendance, Student, Sess).where(Attendance.student_id == Student.id, Attendance.session_id == Sess.id)
    if class_id:
        q = q.where(Sess.class_id == class_id)
    if day:
        try:
            d = date.fromisoformat(day)
            q = q.where(Sess.session_date == d)
        except Exception:
            pass
    rows = db.exec(q).all()
    return templates.TemplateResponse("admin_logs.html", {"request": request, "classes": classes, "rows": rows, "class_id": class_id, "day": day})

@app.get("/admin/export")
def admin_export(class_id: Optional[int] = None, day: Optional[str] = None, db: Session = Depends(get_db)):
    import pandas as pd
    q = select(Attendance, Student, Sess).where(Attendance.student_id == Student.id, Attendance.session_id == Sess.id)
    if class_id:
        q = q.where(Sess.class_id == class_id)
    if day:
        try:
            d = date.fromisoformat(day)
            q = q.where(Sess.session_date == d)
        except Exception:
            pass
    rows = db.exec(q).all()
    data = []
    for att, stu, ses in rows:
        data.append({
            "class_id": ses.class_id,
            "session_date": ses.session_date.isoformat(),
            "student_id": stu.student_id,
            "name": stu.name,
            "recognized_at": att.recognized_at.isoformat(),
            "confidence": att.confidence,
            "liveness": att.liveness_passed,
        })
    import io
    import pandas as pd
    df = pd.DataFrame(data)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=attendance.csv"})
