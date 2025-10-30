import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import mediapipe as mp

from .config import settings

mp_face_mesh = mp.solutions.face_mesh

@lru_cache(maxsize=1)
def get_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def image_bytes_to_bgr(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def embed_image_bgr(img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    app = get_face_app()
    faces = app.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    if face.det_score < settings.min_detection_score:
        return None
    emb = face.normed_embedding.astype(np.float32)
    return emb, float(face.det_score)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def save_face_thumb(img_bgr: np.ndarray, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img_bgr)
    return out_path


def mediapipe_liveness_heuristic(img_bgr: np.ndarray) -> bool:
    # Basic: ensure full face mesh present and plausible geometry
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        res = mesh.process(rgb)
        if not res.multi_face_landmarks:
            return False
        lm = res.multi_face_landmarks[0]
        # Simple heuristic: eye aspect-like check using selected landmarks
        def dist(i, j):
            xi, yi = lm.landmark[i].x, lm.landmark[i].y
            xj, yj = lm.landmark[j].x, lm.landmark[j].y
            return np.hypot(xi - xj, yi - yj)
        # Left eye indices (approx): 33-133 outer corners, 159-145 vertical
        eye_h = dist(159, 145)
        eye_w = dist(33, 133)
        ear = eye_h / (eye_w + 1e-6)
        return 0.18 < ear < 0.35  # crude plausibility band

