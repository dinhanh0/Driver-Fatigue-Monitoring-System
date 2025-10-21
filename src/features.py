from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import cv2
import mediapipe as mp


FEAT_DIM: int = 4

__all__ = ["FEAT_DIM", "process_frame"]

_mp_fd = None
_mp_fm = None


def _get_detectors():
    global _mp_fd, _mp_fm
    if _mp_fd is None:
        _mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    if _mp_fm is None:
        _mp_fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _mp_fd, _mp_fm



LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH4 = [61, 291, 13, 14]


def _gamma_clahe(bgr: np.ndarray, gamma: float = 1.2, clip: float = 2.0) -> np.ndarray:
    """Low-light normalization: gamma + CLAHE on Y (luma)."""
    x = (bgr.astype(np.float32) / 255.0) ** (1.0 / gamma)
    yuv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(yuv)
    y = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(y)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def _pts(lms, idxs, size):
    return [(lms[i].x * size, lms[i].y * size) for i in idxs]


def _ear(pts: list[tuple[float, float]]) -> float:
    # 6 points: p1..p6
    p1, p2, p3, p4, p5, p6 = [np.array(p, dtype=np.float32) for p in pts]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * h)) if h > 1e-6 else np.nan


def _mar(pts: list[tuple[float, float]]) -> float:
    # 4 points: left, right, top, bottom
    l, r, t, b = [np.array(p, dtype=np.float32) for p in pts]
    h = np.linalg.norm(l - r)
    v = np.linalg.norm(t - b)
    return float(v / h) if h > 1e-6 else np.nan

def process_frame(
    frame_bgr: np.ndarray, out_size: int = 224
) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    if frame_bgr is None or frame_bgr.size == 0:
        return None
    import os, time
    profiling = os.environ.get("PROFILE_PROCESS_FRAME", "0") == "1"
    t0 = time.time() if profiling else None

    fd, fm = _get_detectors()

    # Face detection (MediaPipe expects RGB)
    det = fd.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if not det.detections:
        return None

    # Pick the best detection
    d = max(det.detections, key=lambda x: x.score[0])
    h, w = frame_bgr.shape[:2]
    rb = d.location_data.relative_bounding_box
    x1 = max(int(rb.xmin * w), 0)
    y1 = max(int(rb.ymin * h), 0)
    x2 = min(int((rb.xmin + rb.width) * w), w - 1)
    y2 = min(int((rb.ymin + rb.height) * h), h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    # Crop, resize, and low-light normalization
    crop = frame_bgr[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    crop = _gamma_clahe(crop)

    # Face mesh (landmarks)
    res = fm.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark

    ear = np.nanmean(
        [
            _ear(_pts(lms, LEFT_EYE, out_size)),
            _ear(_pts(lms, RIGHT_EYE, out_size)),
        ]
    )
    mar = _mar(_pts(lms, MOUTH4, out_size))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    illum = float(gray.mean())

    # Image to RGB CHW float32 [0,1]
    img_rgb_chw = crop[:, :, ::-1].transpose(2, 0, 1).astype("float32") / 255.0

    feats = np.array([ear, mar, blur_var, illum], dtype="float32")
    # Guard against NaNs (e.g., degenerate geometry)
    if not np.all(np.isfinite(feats)):
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    if profiling:
        t1 = time.time()
        elapsed = t1 - t0
        # print minimal info; tests can redirect stderr if too verbose
        print(f"[PROFILE] process_frame: {elapsed:.4f}s", flush=True)

    return img_rgb_chw, feats