from pydantic import BaseModel

class Settings(BaseModel):
    embedding_threshold: float = 0.38  # cosine distance threshold (lower is closer)
    liveness_required: bool = True
    min_detection_score: float = 0.6
    data_dir: str = "data"
    face_thumb_dir: str = "data/faces"
    db_path: str = "data/attendance.db"
    log_path: str = "data/app.log"

settings = Settings()
