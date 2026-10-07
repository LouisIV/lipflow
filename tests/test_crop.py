import json

import numpy as np
import pytest

from lipflow.crop import (
    CropConfig, clear_crop_config_cache, get_crop_config, load_crop_config,
    save_crop_config, set_crop_config,
)
from lipflow.face import (
    STABLE_REFERENCE, _LIPS, _NOSE_BASE, _UPPER_LIPS, FaceObs, alignment_anchors, mouth_center,
    mouth_rois,
)
from lipflow.tune import compose_view, overlay_landmarks, plot_signal


@pytest.fixture(autouse=True)
def _reset_crop_config():
    clear_crop_config_cache()
    set_crop_config(CropConfig())
    yield
    clear_crop_config_cache()
    set_crop_config(CropConfig())


@pytest.fixture
def home(tmp_path, monkeypatch):
    from lipflow import dictation, crop as crop_mod
    monkeypatch.setattr(dictation, "SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr(dictation, "HOME", str(tmp_path))
    clear_crop_config_cache()
    yield tmp_path
    clear_crop_config_cache()
    set_crop_config(CropConfig())


def fake_face(lower_extra=0.0):
    """478 MediaPipe-sized points: eyes, nose, upper lips, lower lips."""
    pts = np.zeros((478, 2), np.float32)
    for i in (7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246):
        pts[i] = (120, 100)
    for i in (249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466):
        pts[i] = (200, 100)
    for i in _NOSE_BASE:
        pts[i] = (160, 145)
    for i in _LIPS:
        pts[i] = (160, 190)
    for i in _UPPER_LIPS:
        pts[i] = (160, 175)
    for i in set(_LIPS) - set(_UPPER_LIPS):
        pts[i] = (160, 205 + lower_extra)
    pts[61] = (125, 185)
    pts[291] = (195, 185)
    pts[13], pts[14] = (160, 178), (160, 200)
    return pts


def textured_frames(n=5, h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w), dtype=np.uint8) for _ in range(n)]


def _legacy_mouth_rois(gray_frames, anchors, crop=96, window_margin=12):
    """The pre-tuner crop loop, kept here so defaults cannot drift."""
    import cv2
    from lipflow.face import _interpolate
    lms = _interpolate(anchors)
    n = len(lms)
    half = crop // 2
    patches = []
    for i, frame in enumerate(gray_frames):
        offset = None
        if isinstance(frame, tuple):
            frame, offset = frame
        m = min(window_margin // 2, i, n - 1 - i)
        smoothed = np.mean(lms[i - m:i + m + 1], axis=0)
        smoothed += lms[i].mean(axis=0) - smoothed.mean(axis=0)
        tf, _ = cv2.estimateAffinePartial2D(smoothed.astype(np.float32), STABLE_REFERENCE.astype(np.float32),
                                            method=cv2.LMEDS)
        if tf is None:
            tf = cv2.estimateAffinePartial2D(lms[i].astype(np.float32), STABLE_REFERENCE.astype(np.float32))[0]
        mouth = smoothed[3] @ tf[:, :2].T + tf[:, 2]
        if offset is not None:
            tf = tf.copy()
            tf[:, 2] += tf[:, :2] @ np.asarray(offset, dtype=tf.dtype)
        warped = cv2.warpAffine(frame, tf, (256, 256), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cx = int(round(np.clip(mouth[0], half, 256 - half)))
        cy = int(round(np.clip(mouth[1], half, 256 - half)))
        patches.append(warped[cy - half:cy + half, cx - half:cx + half])
    return np.stack(patches)


def test_defaults_match_legacy_mouth_rois():
    frames = textured_frames()
    anchors = [STABLE_REFERENCE * 1.5 + 80 for _ in frames]
    a = mouth_rois(frames, anchors)
    b = mouth_rois(frames, anchors, cfg=CropConfig())
    legacy = _legacy_mouth_rois(frames, anchors)
    assert a.shape == (5, 96, 96) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert np.array_equal(a, legacy)


def test_offset_shifts_the_crop():
    frames = textured_frames()
    anchors = [STABLE_REFERENCE * 1.8 + (100, 60) for _ in frames]
    base = mouth_rois(frames, anchors, cfg=CropConfig())
    up = mouth_rois(frames, anchors, cfg=CropConfig(mouth_offset_y=-24))
    down = mouth_rois(frames, anchors, cfg=CropConfig(mouth_offset_y=24))
    assert not np.array_equal(base, up)
    assert not np.array_equal(up, down)


def test_scale_changes_the_crop():
    frames = textured_frames()
    anchors = [STABLE_REFERENCE * 1.8 + (100, 60) for _ in frames]
    base = mouth_rois(frames, anchors, cfg=CropConfig())
    tight = mouth_rois(frames, anchors, cfg=CropConfig(crop_scale=0.6))
    wide = mouth_rois(frames, anchors, cfg=CropConfig(crop_scale=1.6))
    assert tight.shape == base.shape == wide.shape
    assert not np.array_equal(base, tight)
    assert not np.array_equal(base, wide)


def test_clahe_and_brightness_change_pixels_not_shape():
    frames = textured_frames()
    anchors = [STABLE_REFERENCE * 1.5 + 50 for _ in frames]
    base = mouth_rois(frames, anchors, cfg=CropConfig())
    clahe = mouth_rois(frames, anchors, cfg=CropConfig(clahe=True))
    bright = mouth_rois(frames, anchors, cfg=CropConfig(brightness_norm=True))
    assert clahe.shape == bright.shape == base.shape
    assert not np.array_equal(base, clahe)
    assert not np.array_equal(base, bright)
    assert abs(bright.mean() - 0.421 * 255) < abs(base.mean() - 0.421 * 255) + 5


def test_align_off_still_returns_96():
    frames = textured_frames(3)
    anchors = [STABLE_REFERENCE * 2.0 + (80, 40) for _ in frames]
    rois = mouth_rois(frames, anchors, cfg=CropConfig(align=False))
    assert rois.shape == (3, 96, 96)


def test_lips_anchor_matches_mean_of_lip_points():
    pts = fake_face()
    got = mouth_center(pts, "lips")
    assert np.allclose(got, pts[_LIPS].mean(0))
    a = alignment_anchors(pts, "lips")
    assert a.shape == (4, 2)
    assert np.allclose(a[3], got)


def test_upper_anchor_sits_above_all_lips_mean():
    pts = fake_face()
    assert mouth_center(pts, "upper")[1] < mouth_center(pts, "lips")[1]


def test_nose_anchor_ignores_pulled_down_lips():
    a = fake_face(0)
    b = fake_face(40)  # beard dragging lower-lip landmarks down
    assert mouth_center(b, "lips")[1] > mouth_center(a, "lips")[1] + 5
    assert np.allclose(mouth_center(a, "nose"), mouth_center(b, "nose"), atol=1e-5)


def test_faceobs_anchors_follow_config():
    set_crop_config(CropConfig(mouth_anchor="upper"))
    pts = fake_face()
    assert np.allclose(FaceObs(pts).anchors[3], mouth_center(pts, "upper"))
    set_crop_config(CropConfig())


def test_from_dict_fills_defaults_and_rejects_junk():
    cfg = CropConfig.from_dict({"mouth_anchor": "beard", "crop_scale": 99, "nope": 1, "clahe": "on"})
    assert cfg.mouth_anchor == "lips"
    assert cfg.crop_scale == 2.0
    assert cfg.clahe is True
    assert cfg.align is True
    assert cfg.smooth == 12


def test_roundtrip_to_dict():
    cfg = CropConfig(mouth_anchor="upper", mouth_offset_y=-8, crop_scale=0.85, clahe=True)
    assert CropConfig.from_dict(cfg.to_dict()) == cfg


def test_save_load_and_reset(home):
    cfg = CropConfig(mouth_anchor="nose", crop_scale=0.9, mouth_offset_y=-12)
    path = save_crop_config(cfg)
    assert path == str(home / "settings.json")
    data = json.loads((home / "settings.json").read_text())
    assert data["crop"]["mouth_anchor"] == "nose"
    clear_crop_config_cache()
    loaded = load_crop_config()
    assert loaded.mouth_anchor == "nose" and loaded.crop_scale == pytest.approx(0.9)
    save_crop_config(CropConfig())
    data = json.loads((home / "settings.json").read_text())
    assert "crop" not in data
    clear_crop_config_cache()
    assert load_crop_config() == CropConfig()


def test_save_does_not_clobber_other_settings(home):
    from lipflow.dictation import save_settings
    save_settings({"camera": "auto", "whisper": True})
    save_crop_config(CropConfig(smooth=6))
    data = json.loads((home / "settings.json").read_text())
    assert data["whisper"] is True and data["crop"]["smooth"] == 6


def test_missing_settings_are_defaults(home):
    clear_crop_config_cache()
    assert get_crop_config() == CropConfig()


def test_overlay_and_compose_do_not_crash():
    bgr = np.zeros((240, 320, 3), np.uint8)
    obs = FaceObs(fake_face())
    quad = np.float32([[100, 140], [180, 140], [180, 200], [100, 200]])
    drawn = overlay_landmarks(bgr, obs, quad)
    assert drawn.shape == bgr.shape
    none = overlay_landmarks(bgr, None, None)
    assert none.shape == bgr.shape
    crop = np.arange(96 * 96, dtype=np.uint8).reshape(96, 96)
    canvas = compose_view(drawn, crop, dict(score=0.9, dropout=0.1, jitter=1.5,
                                            anchor="upper", scale=0.9, offset=-8),
                          [0.05, 0.1, 0.2, 0.12])
    assert canvas.ndim == 3 and canvas.shape[1] > 320
    assert plot_signal([0.1, 0.2, 0.15], 200, 40, 0, 1).shape == (40, 200, 3)


def test_tune_subcommand_exists(monkeypatch):
    monkeypatch.setenv("DISPLAY", "")  # Linux CI: pynput must not be imported just to parse --help
    from lipflow.__main__ import main
    with pytest.raises(SystemExit) as ei:
        main(["tune", "-h"])
    assert ei.value.code == 0
