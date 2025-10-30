from datetime import datetime, date, time
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship

class Class(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    code: str

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    student_id: str
    class_id: Optional[int] = Field(default=None, foreign_key="class.id")
    embedding_json: str  # JSON list of floats
    face_thumb_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Session(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    class_id: int = Field(foreign_key="class.id")
    session_date: date
    start_time: time
    end_time: Optional[time] = None

class Attendance(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="session.id")
    student_id: int = Field(foreign_key="student.id")
    recognized_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float
    liveness_passed: bool = True
    source: str = "webcam"
