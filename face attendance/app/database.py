import os
from sqlmodel import SQLModel, create_engine
from .config import settings

os.makedirs(settings.data_dir, exist_ok=True)

engine = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})

def init_db():
    SQLModel.metadata.create_all(engine)
