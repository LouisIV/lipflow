"""Live mouth-crop tuner: `lipflow tune`.

Opens the webcam and shows the raw frame (landmarks + the crop rectangle) next to
the exact 96×96 the VSR model receives, with sliders for CropConfig. **s** saves
into settings.json so dictation uses the same values; **r** resets to defaults.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from .camera import PINK, PINK_SOFT, open_capture
from .crop import ANCHOR_MODES, CropConfig, get_crop_config, load_crop_config, save_crop_config, set_crop_config
from .face import FaceTracker, alignment_anchors, face_box, mouth_rois_meta
from .paths import WHO

VIEW, KNOBS = "Lipflow tune", "Lipflow knobs"
HIST = 150
BUF = 24
LEFT_W, LEFT_H = 640, 360
CROP_SHOW = 320
PLOT_H = 88
ACCENT = PINK
GREEN = (90, 200, 140)
AMBER = (40, 180, 255)
WHITE = (235, 235, 235)
DIM = (170, 170, 170)


def _put(img, text, xy, scale=0.48, color=WHITE, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay_landmarks(bgr: np.ndarray, obs, quad: np.ndarray | None) -> np.ndarray:
    """Draw face box, lip contours, the 4 alignment anchors, and the model-crop quad."""
    view = bgr.copy()
    if obs is None:
        _put(view, "looking for your face...", (16, view.shape[0] - 18), 0.55, DIM)
        return view
    x0, y0, x1, y1 = face_box(obs.pts)
    cv2.rectangle(view, (x0, y0), (x1, y1), DIM, 1, cv2.LINE_AA)
    overlay = view.copy()
    for contour in (obs.outer_lips, obs.inner_lips):
        cv2.polylines(overlay, [np.round(contour).astype(np.int32)], True, PINK_SOFT, 1, cv2.LINE_AA)
    view = cv2.addWeighted(overlay, 0.65, view, 0.35, 0)
    for x, y in obs.lip_points:
        cv2.circle(view, (int(round(x)), int(round(y))), 2, ACCENT, -1, cv2.LINE_AA)
    colors = ((80, 180, 255), (80, 220, 80), (220, 180, 60), (80, 80, 255))  # eyes, nose, mouth
    labels = ("R eye", "L eye", "nose", "mouth")
    for (x, y), c, lab in zip(obs.anchors, colors, labels):
        p = (int(round(x)), int(round(y)))
        cv2.circle(view, p, 5, c, -1, cv2.LINE_AA)
        _put(view, lab, (p[0] + 6, p[1] - 6), 0.4, c)
    if quad is not None and len(quad) >= 4:
        pts = np.round(quad).astype(np.int32)
        cv2.polylines(view, [pts], True, ACCENT, 2, cv2.LINE_AA)
    return view


def plot_signal(values: list[float], w: int, h: int, lo: float, hi: float, color=ACCENT) -> np.ndarray:
    img = np.full((h, w, 3), 18, np.uint8)
    cv2.line(img, (0, h // 2), (w, h // 2), (40, 40, 40), 1)
    if len(values) < 2:
        return img
    span = (hi - lo) or 1.0
    xs = np.linspace(0, w - 1, len(values))
    pts = []
    for x, v in zip(xs, values):
        y = int(np.clip(h - 1 - (float(v) - lo) / span * (h - 2), 0, h - 1))
        pts.append((int(x), y))
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, 1, cv2.LINE_AA)
    return img


def compose_view(camera_bgr: np.ndarray, crop_gray: np.ndarray | None, stats: dict,
                 opens: list[float], flash: str = "") -> np.ndarray:
    """Side-by-side camera + model crop, with live signals underneath."""
    cam = cv2.resize(camera_bgr, (LEFT_W, LEFT_H), interpolation=cv2.INTER_AREA)
    cam = cv2.flip(cam, 1)  # selfie view, matching the dictation HUD
    if crop_gray is None:
        crop = np.full((CROP_SHOW, CROP_SHOW, 3), 28, np.uint8)
        _put(crop, "no face", (20, CROP_SHOW // 2), 0.7, DIM)
    else:
        g = cv2.resize(crop_gray, (CROP_SHOW, CROP_SHOW), interpolation=cv2.INTER_NEAREST)
        crop = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    gap = 10
    top = np.full((LEFT_H, LEFT_W + gap + CROP_SHOW, 3), 12, np.uint8)
    top[:, :LEFT_W] = cam
    y0 = (LEFT_H - CROP_SHOW) // 2
    top[y0:y0 + CROP_SHOW, LEFT_W + gap:] = crop
    _put(top, "camera + landmarks  (mirrored)", (12, 22), 0.5, DIM)
    _put(top, "model crop  (what VSR sees)", (LEFT_W + gap + 8, y0 + 18), 0.5, DIM)

    panel_w = top.shape[1]
    plot = plot_signal(list(opens), panel_w, PLOT_H, 0.0, 0.55)
    _put(plot, "mouth open", (8, 16), 0.42, DIM)
    if opens:
        _put(plot, f"{opens[-1]:.3f}", (panel_w - 70, 16), 0.42, ACCENT)

    bar = np.full((52, panel_w, 3), 16, np.uint8)
    drop = stats.get("dropout", 1.0)
    jitter = stats.get("jitter", 0.0)
    score = stats.get("score", 0.0)
    line = (f"score {score:.2f}   dropout {drop:.0%}   jitter {jitter:.1f}px   "
            f"anchor {stats.get('anchor', 'lips')}   scale {stats.get('scale', 1):.2f}   "
            f"offsetY {stats.get('offset', 0):+.0f}")
    _put(bar, line, (12, 22), 0.48, WHITE)
    hint = flash or "s save to Lipflow settings    r reset defaults    q quit"
    _put(bar, hint, (12, 42), 0.42, GREEN if flash else DIM)

    return np.vstack([top, plot, bar])


def knobs_legend() -> np.ndarray:
    img = np.full((150, 520, 3), 24, np.uint8)
    lines = [
        "anchor:  0 lips (default)  1 outer  2 inner  3 upper  4 nose",
        "offsetY: 40 = 0     drag left = shift crop up (away from beard)",
        "scale:   100 = default 96px of aligned face     <100 tighter",
        "smooth:  12 = default landmark window     0 = off",
        "align 1 / CLAHE 0 / bright 0  are on/off. Save with s.",
    ]
    for i, line in enumerate(lines):
        _put(img, line, (10, 28 + i * 24), 0.42, WHITE)
    return img


def _cfg_from_trackbars() -> CropConfig:
    def g(name):
        return cv2.getTrackbarPos(name, KNOBS)
    return CropConfig(
        detect_conf=max(g("detect%"), 5) / 100.0,
        track_conf=max(g("track%"), 5) / 100.0,
        mouth_anchor=ANCHOR_MODES[min(g("anchor"), len(ANCHOR_MODES) - 1)],
        mouth_offset_y=float(g("offsetY") - 40),
        crop_scale=max(g("scale%"), 50) / 100.0,
        align=bool(g("align")),
        smooth=g("smooth"),
        clahe=bool(g("clahe")),
        brightness_norm=bool(g("bright")),
    )


def _set_trackbars(cfg: CropConfig):
    cv2.setTrackbarPos("detect%", KNOBS, int(round(cfg.detect_conf * 100)))
    cv2.setTrackbarPos("track%", KNOBS, int(round(cfg.track_conf * 100)))
    cv2.setTrackbarPos("anchor", KNOBS, ANCHOR_MODES.index(cfg.mouth_anchor))
    cv2.setTrackbarPos("offsetY", KNOBS, int(round(cfg.mouth_offset_y + 40)))
    cv2.setTrackbarPos("scale%", KNOBS, int(round(cfg.crop_scale * 100)))
    cv2.setTrackbarPos("align", KNOBS, 1 if cfg.align else 0)
    cv2.setTrackbarPos("smooth", KNOBS, cfg.smooth)
    cv2.setTrackbarPos("clahe", KNOBS, 1 if cfg.clahe else 0)
    cv2.setTrackbarPos("bright", KNOBS, 1 if cfg.brightness_norm else 0)


def _make_knobs(cfg: CropConfig):
    cv2.namedWindow(KNOBS, cv2.WINDOW_NORMAL)
    dummy = lambda *_: None
    cv2.createTrackbar("detect%", KNOBS, 50, 90, dummy)
    cv2.createTrackbar("track%", KNOBS, 50, 90, dummy)
    cv2.createTrackbar("anchor", KNOBS, 0, len(ANCHOR_MODES) - 1, dummy)
    cv2.createTrackbar("offsetY", KNOBS, 40, 80, dummy)
    cv2.createTrackbar("scale%", KNOBS, 100, 200, dummy)
    cv2.createTrackbar("align", KNOBS, 1, 1, dummy)
    cv2.createTrackbar("smooth", KNOBS, 12, 24, dummy)
    cv2.createTrackbar("clahe", KNOBS, 0, 1, dummy)
    cv2.createTrackbar("bright", KNOBS, 0, 1, dummy)
    _set_trackbars(cfg)
    cv2.imshow(KNOBS, knobs_legend())


def _request_camera():
    """Prompt for camera access on the main thread (macOS)."""
    os.environ.pop("OPENCV_AVFOUNDATION_SKIP_AUTH", None)
    if sys.platform != "darwin":
        return
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
        status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
        if status == 0:
            done = threading.Event()
            AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                AVMediaTypeVideo, lambda granted: done.set())
            done.wait(timeout=120)
        elif status in (1, 2):
            print(f"Camera access denied. Enable {WHO} in System Settings → Privacy & Security → Camera.")
    except Exception:
        pass


def run(camera: "int | str" = "auto") -> int:
    load_crop_config()
    cfg = get_crop_config()
    _request_camera()
    try:
        cap, file_fps = open_capture(camera)
    except Exception as e:
        print(f"[tune] {e}")
        return 1
    try:
        cv2.namedWindow(VIEW, cv2.WINDOW_NORMAL)
        cv2.imshow(VIEW, np.zeros((LEFT_H + PLOT_H + 52, LEFT_W + 10 + CROP_SHOW, 3), np.uint8))
        if cv2.waitKey(1) < -1:
            raise cv2.error("no display")
    except cv2.error:
        print("[tune] OpenCV cannot open a window. The tuner needs a screen (not a headless SSH).")
        cap.release()
        return 1
    _make_knobs(cfg)

    tracker = FaceTracker(detect_conf=cfg.detect_conf, track_conf=cfg.track_conf)
    t0 = time.time()
    n_read = 0
    grays, pts_buf = deque(maxlen=BUF), deque(maxlen=BUF)
    opens: deque[float] = deque(maxlen=HIST)
    hits: deque[float] = deque(maxlen=HIST)
    scores: deque[float] = deque(maxlen=HIST)
    last_center = None
    jitter = 0.0
    flash, flash_until = "", 0.0
    last_conf = (cfg.detect_conf, cfg.track_conf)

    print("Lipflow mouth-crop tuner")
    print("  s save   r reset to defaults   q quit")
    print(f"  current: {cfg.to_dict()}")
    try:
        while True:
            ok, frame = cap.read()
            if file_fps:
                n_read += 1
                time.sleep(max(0.0, t0 + n_read / file_fps - time.time()))
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    n_read, t0 = 0, time.time()
                    continue
            elif not ok:
                time.sleep(0.01)
                continue
            cfg = _cfg_from_trackbars()
            set_crop_config(cfg)
            conf = (cfg.detect_conf, cfg.track_conf)
            if conf != last_conf:
                tracker.close()
                tracker = FaceTracker(detect_conf=cfg.detect_conf, track_conf=cfg.track_conf)
                last_conf = conf

            now = time.time()
            obs = tracker.detect(frame, int((now - t0) * 1000))
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            pts_buf.append(None if obs is None else obs.pts)
            hits.append(1.0 if obs else 0.0)
            scores.append(obs.score if obs else 0.0)
            opens.append(obs.mouth_open if obs else 0.0)

            anchors = [None if p is None else alignment_anchors(p, cfg.mouth_anchor) for p in pts_buf]
            patches, metas = mouth_rois_meta(list(grays), anchors, cfg=cfg, causal=True)
            crop = quad = None
            if patches:
                crop = patches[-1]
                quad = metas[-1]["quad"]
                center = metas[-1]["center"]
                if last_center is not None:
                    jitter = float(np.linalg.norm(center - last_center))
                last_center = center
            else:
                last_center = None

            drawn = overlay_landmarks(frame, obs, quad)
            n = max(len(hits), 1)
            stats = dict(score=float(sum(scores) / max(len(scores), 1)),
                         dropout=1.0 - float(sum(hits) / n),
                         jitter=jitter, anchor=cfg.mouth_anchor,
                         scale=cfg.crop_scale, offset=cfg.mouth_offset_y)
            msg = flash if now < flash_until else ""
            canvas = compose_view(drawn, crop, stats, list(opens), msg)
            cv2.imshow(VIEW, canvas)
            cv2.imshow(KNOBS, knobs_legend())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_crop_config(cfg)
                flash, flash_until = f"saved {path}", now + 2.5
                print(f"[tune] saved {cfg.to_dict()} -> {path}")
            if key == ord("r"):
                cfg = CropConfig()
                set_crop_config(cfg)
                _set_trackbars(cfg)
                save_crop_config(cfg)
                flash, flash_until = "reset to defaults (and saved)", now + 2.5
                print("[tune] reset to defaults")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0
