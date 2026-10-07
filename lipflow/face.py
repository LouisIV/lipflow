"""Face tracking and mouth-ROI extraction.

The VSR model was trained on 96x96 grayscale mouth crops taken from faces that were
first aligned to a mean face (eyes, nose base, mouth centre). We reproduce that exact
preprocessing, but compute the four anchor points live with MediaPipe FaceLandmarker
so a recording is ready for inference the moment you stop talking.

Crop geometry (scale, vertical offset, which landmarks anchor the mouth, alignment,
smoothing, CLAHE / brightness) is `CropConfig` in crop.py, saved from `lipflow tune`.
"""
from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .crop import TRAIN_GRAY_MEAN, CropConfig, get_crop_config

HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "face_landmarker.task")

# Subject's right eye is on the image left in an unmirrored camera frame, matching
# dlib points 36-41 of the reference face.
_RIGHT_EYE = [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]
_LEFT_EYE = [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]
_NOSE_BASE = [97, 98, 2, 326, 327]  # ~ dlib 31-35
_LIPS = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
         185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402,
         405, 409, 415]
_OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
               181, 91, 146]
_INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
# Outer + inner upper lip, including the corners; no chin / lower-lip points a beard can pull.
_UPPER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
               78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
_UPPER_INNER, _LOWER_INNER = 13, 14


def _stable_reference(size=256):
    ref = np.load(os.path.join(HERE, "mean_face.npy"))
    pts = np.vstack([ref[36:42].mean(0), ref[42:48].mean(0), ref[31:36].mean(0), ref[48:68].mean(0)])
    return pts - (256 - size) / 2.0


STABLE_REFERENCE = _stable_reference()


def _landmark_score(landmarks) -> float:
    """Mean presence (or visibility) when MediaPipe fills it in; 1.0 if unset."""
    vals = []
    for p in landmarks:
        for attr in ("presence", "visibility"):
            v = getattr(p, attr, None)
            if v is not None and float(v) > 0:
                vals.append(float(v))
                break
    return float(sum(vals) / len(vals)) if vals else 1.0


def _mouth_from_nose(pts: np.ndarray) -> np.ndarray:
    """Mouth centre predicted from eyes + nose, ignoring lip/chin landmarks.

    Uses the mean-face ratio |mouth − nose| / |nose − mid-eyes| along the face's
    down vector, so a beard that drags the lower lip points cannot pull the crop.
    """
    r, l = pts[_RIGHT_EYE].mean(0), pts[_LEFT_EYE].mean(0)
    nose = pts[_NOSE_BASE].mean(0)
    down = nose - (r + l) / 2
    ref = STABLE_REFERENCE
    ref_down = ref[2] - (ref[0] + ref[1]) / 2
    k = np.linalg.norm(ref[3] - ref[2]) / (np.linalg.norm(ref_down) + 1e-6)
    return nose + down * k


def mouth_center(pts: np.ndarray, mode: str = "lips") -> np.ndarray:
    """2-vector mouth point in image coordinates for the 4th alignment / crop anchor."""
    if mode == "outer":
        return pts[_OUTER_LIPS].mean(0)
    if mode == "inner":
        return pts[_INNER_LIPS].mean(0)
    if mode == "upper":
        return pts[_UPPER_LIPS].mean(0)
    if mode == "nose":
        return _mouth_from_nose(pts)
    return pts[_LIPS].mean(0)


def alignment_anchors(pts: np.ndarray, mode: str = "lips") -> np.ndarray:
    """4×2: right eye, left eye, nose base, mouth centre (image coords)."""
    return np.vstack([
        pts[_RIGHT_EYE].mean(0), pts[_LEFT_EYE].mean(0),
        pts[_NOSE_BASE].mean(0), mouth_center(pts, mode),
    ])


def face_box(pts: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return int(x0), int(y0), int(x1), int(y1)


class FaceTracker:
    """Per-frame landmarks. Use VIDEO mode so MediaPipe tracks between frames."""

    def __init__(self, model_path: str = DEFAULT_MODEL, *,
                 detect_conf: float | None = None, track_conf: float | None = None):
        cfg = get_crop_config()
        if detect_conf is None:
            detect_conf = cfg.detect_conf
        if track_conf is None:
            track_conf = cfg.track_conf
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.abspath(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=float(detect_conf),
            min_tracking_confidence=float(track_conf),
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1
        self.detect_conf = float(detect_conf)
        self.track_conf = float(track_conf)

    def detect(self, frame_bgr: np.ndarray, ts_ms: int) -> "FaceObs | None":
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing time
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lm = res.face_landmarks[0]
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        return FaceObs(pts, _landmark_score(lm))

    def close(self):
        self._lm.close()


class FaceObs:
    __slots__ = ("pts", "score")

    def __init__(self, pts: np.ndarray, score: float = 1.0):
        self.pts = pts
        self.score = float(score)

    @property
    def anchors(self) -> np.ndarray:
        """4x2: right eye, left eye, nose base, mouth centre (image coords)."""
        return alignment_anchors(self.pts, get_crop_config().mouth_anchor)

    @property
    def mouth_open(self) -> float:
        """Inner-lip gap normalised by mouth width — a cheap 'is the mouth moving' signal."""
        p = self.pts
        width = np.linalg.norm(p[61] - p[291]) + 1e-6
        return float(np.linalg.norm(p[_UPPER_INNER] - p[_LOWER_INNER]) / width)

    @property
    def outer_lips(self) -> np.ndarray:
        return self.pts[_OUTER_LIPS]

    @property
    def inner_lips(self) -> np.ndarray:
        return self.pts[_INNER_LIPS]

    @property
    def lip_points(self) -> np.ndarray:
        return self.pts[_LIPS]


def _interpolate(anchors: list["np.ndarray | None"]) -> "list[np.ndarray] | None":
    valid = [i for i, a in enumerate(anchors) if a is not None]
    if not valid:
        return None
    out = list(anchors)
    for a, b in zip(valid, valid[1:]):
        for k in range(1, b - a):
            out[a + k] = out[a] + (out[b] - out[a]) * (k / (b - a))
    for i in range(valid[0]):
        out[i] = out[valid[0]]
    for i in range(valid[-1] + 1, len(out)):
        out[i] = out[valid[-1]]
    return out


def _crop_side(cfg: CropConfig) -> int:
    side = int(round(cfg.crop * cfg.crop_scale))
    side = min(max(side, 8), 256)
    if side % 2:
        side = min(side + 1, 256)
        if side % 2:
            side -= 1
    return max(side, 8)


def _postprocess(patch: np.ndarray, cfg: CropConfig) -> np.ndarray:
    if patch is None or patch.size == 0 or min(patch.shape[:2]) < 2:
        return np.zeros((cfg.crop, cfg.crop), np.uint8)
    if patch.shape[0] != cfg.crop or patch.shape[1] != cfg.crop:
        patch = cv2.resize(patch, (cfg.crop, cfg.crop), interpolation=cv2.INTER_LINEAR)
    if cfg.clahe:
        patch = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(patch)
    if cfg.brightness_norm:
        patch = np.clip(patch.astype(np.float32) - float(patch.mean()) + TRAIN_GRAY_MEAN, 0, 255).astype(np.uint8)
    return patch


def _inverse_quad(tf: np.ndarray, cx: float, cy: float, half: float) -> np.ndarray:
    """Crop rectangle corners in the coordinate system `tf` maps from."""
    corners = np.float32([[cx - half, cy - half], [cx + half, cy - half],
                          [cx + half, cy + half], [cx - half, cy + half]])
    try:
        inv = cv2.invertAffineTransform(tf)
    except cv2.error:
        return corners
    return cv2.transform(corners.reshape(-1, 1, 2), inv).reshape(-1, 2)


def _patch_aligned(frame: np.ndarray, smoothed: np.ndarray, fallback: np.ndarray,
                   cfg: CropConfig, offset) -> tuple[np.ndarray, dict]:
    tf, _ = cv2.estimateAffinePartial2D(smoothed.astype(np.float32), STABLE_REFERENCE.astype(np.float32),
                                        method=cv2.LMEDS)
    if tf is None:
        tf = cv2.estimateAffinePartial2D(fallback.astype(np.float32), STABLE_REFERENCE.astype(np.float32))[0]
    mouth = smoothed[3] @ tf[:, :2].T + tf[:, 2]  # landmarks are in full-frame coordinates
    mouth = np.asarray(mouth, dtype=np.float64)
    mouth[1] += cfg.mouth_offset_y
    tf_full = tf
    if offset is not None:  # the image is a crop: shift the translation to crop pixels
        tf = tf.copy()
        tf[:, 2] += tf[:, :2] @ np.asarray(offset, dtype=tf.dtype)
    warped = cv2.warpAffine(frame, tf, (256, 256), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    side = _crop_side(cfg)
    half = side / 2.0
    cx = int(round(np.clip(mouth[0], half, 256 - half)))
    cy = int(round(np.clip(mouth[1], half, 256 - half)))
    h = side // 2
    patch = warped[cy - h:cy + h, cx - h:cx + h]
    quad = _inverse_quad(tf_full, cx, cy, h)
    return _postprocess(patch, cfg), {"center": np.array([cx, cy], np.float32), "quad": quad, "tf": tf_full}


def _patch_unaligned(frame: np.ndarray, smoothed: np.ndarray, cfg: CropConfig,
                     offset) -> tuple[np.ndarray, dict]:
    """Skip the mean-face warp: square crop around the mouth in image pixels, then resize to 96."""
    ref_eye = np.linalg.norm(STABLE_REFERENCE[0] - STABLE_REFERENCE[1]) + 1e-6
    img_eye = np.linalg.norm(smoothed[0] - smoothed[1])
    px = img_eye / ref_eye
    side = int(round(_crop_side(cfg) * px))
    if side % 2:
        side += 1
    side = max(side, 8)
    half = side / 2.0
    mouth = np.asarray(smoothed[3], dtype=np.float64)
    mouth[1] += cfg.mouth_offset_y * px
    if offset is not None:
        mouth = mouth - np.asarray(offset, dtype=np.float64)
    h, w = frame.shape[:2]
    cx = float(np.clip(mouth[0], half, max(w - half, half)))
    cy = float(np.clip(mouth[1], half, max(h - half, half)))
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + side, y0 + side
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        patch = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    else:
        patch = frame[y0:y1, x0:x1]
    ox, oy = (offset if offset is not None else (0, 0))
    quad = np.float32([[x0 + ox, y0 + oy], [x1 + ox, y0 + oy], [x1 + ox, y1 + oy], [x0 + ox, y1 + oy]])
    return _postprocess(patch, cfg), {"center": np.array([cx + ox, cy + oy], np.float32), "quad": quad, "tf": None}


def mouth_rois(gray_frames: list[np.ndarray], anchors: list["np.ndarray | None"],
               crop: int = 96, window_margin: int = 12,
               cfg: CropConfig | None = None, causal: bool = False) -> "np.ndarray | None":
    """Align each frame to the mean face and cut a crop x crop patch around the mouth.

    Mirrors Auto-AVSR's VideoProcess: temporally smoothed landmarks -> similarity
    transform onto the reference -> fixed-size patch centred on the mouth.
    """
    patches, _ = mouth_rois_meta(gray_frames, anchors, crop, window_margin, cfg, causal=causal)
    return None if patches is None else np.stack(patches)


def mouth_rois_meta(gray_frames: list[np.ndarray], anchors: list["np.ndarray | None"],
                    crop: int = 96, window_margin: int = 12,
                    cfg: CropConfig | None = None, causal: bool = False) -> tuple:
    """Like mouth_rois, plus per-frame `{center, quad, tf}` for the tuner overlay.

    `quad` is the crop rectangle in full-frame coordinates.
    """
    if cfg is None:
        cfg = CropConfig(crop=crop, smooth=window_margin)
    lms = _interpolate(anchors)
    if lms is None:
        return None, []
    n = len(lms)
    patches, metas = [], []
    for i, frame in enumerate(gray_frames):
        offset = None
        if isinstance(frame, tuple):  # (face crop, (x0, y0)): the capture keeps only the face region
            frame, offset = frame
        m = min(cfg.smooth // 2, i) if causal else min(cfg.smooth // 2, i, n - 1 - i)
        smoothed = np.mean(lms[i - m:i + m + 1], axis=0)
        smoothed += lms[i].mean(axis=0) - smoothed.mean(axis=0)
        if cfg.align:
            patch, meta = _patch_aligned(frame, smoothed, lms[i], cfg, offset)
        else:
            patch, meta = _patch_unaligned(frame, smoothed, cfg, offset)
        patches.append(patch)
        metas.append(meta)
    return patches, metas
