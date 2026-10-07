"""Mouth-crop knobs used by the VSR pipeline and the `lipflow tune` tool.

Saved under the `crop` key of settings.json (same file as the rest of Lipflow's
settings). Missing or empty `crop` → the original 96×96 aligned crop, so anyone
who never tunes sees unchanged behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

ANCHOR_MODES = ("lips", "outer", "inner", "upper", "nose")

# Auto-AVSR's training mean (vsr.py _MEAN) as a uint8 gray level, used by brightness_norm.
TRAIN_GRAY_MEAN = 0.421 * 255.0


def _f(v, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _i(v, default: int, lo: int, hi: int) -> int:
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def _b(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


@dataclass(frozen=True)
class CropConfig:
    """Parameters that change how a frame becomes the 96×96 the model reads."""
    detect_conf: float = 0.5
    track_conf: float = 0.5
    mouth_anchor: str = "lips"
    mouth_offset_y: float = 0.0   # pixels in the 256 aligned face; + is down
    crop_scale: float = 1.0       # 1 = 96px of the aligned face; >1 zooms out
    align: bool = True
    smooth: int = 12              # bidirectional landmark window (was window_margin)
    clahe: bool = False
    brightness_norm: bool = False
    crop: int = 96                # model input; not a tuner knob

    def to_dict(self) -> dict:
        return {
            "detect_conf": self.detect_conf,
            "track_conf": self.track_conf,
            "mouth_anchor": self.mouth_anchor,
            "mouth_offset_y": self.mouth_offset_y,
            "crop_scale": self.crop_scale,
            "align": self.align,
            "smooth": self.smooth,
            "clahe": self.clahe,
            "brightness_norm": self.brightness_norm,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "CropConfig":
        d = d or {}
        anchor = d.get("mouth_anchor", "lips")
        if anchor not in ANCHOR_MODES:
            anchor = "lips"
        return cls(
            detect_conf=_f(d.get("detect_conf", 0.5), 0.5, 0.05, 0.95),
            track_conf=_f(d.get("track_conf", 0.5), 0.5, 0.05, 0.95),
            mouth_anchor=anchor,
            mouth_offset_y=_f(d.get("mouth_offset_y", 0.0), 0.0, -80.0, 80.0),
            crop_scale=_f(d.get("crop_scale", 1.0), 1.0, 0.5, 2.0),
            align=_b(d.get("align", True), True),
            smooth=_i(d.get("smooth", 12), 12, 0, 24),
            clahe=_b(d.get("clahe", False), False),
            brightness_norm=_b(d.get("brightness_norm", False), False),
        )


_cache: CropConfig | None = None


def get_crop_config() -> CropConfig:
    global _cache
    if _cache is None:
        _cache = load_crop_config()
    return _cache


def set_crop_config(cfg: CropConfig) -> None:
    global _cache
    _cache = cfg


def load_crop_config() -> CropConfig:
    from .dictation import load_settings
    cfg = CropConfig.from_dict(load_settings().get("crop"))
    set_crop_config(cfg)
    return cfg


def save_crop_config(cfg: CropConfig | None = None) -> str:
    """Write `cfg` into settings.json. Defaults are stored as a missing `crop` key."""
    from .dictation import SETTINGS, load_settings, save_settings
    cfg = CropConfig() if cfg is None else cfg
    set_crop_config(cfg)
    d = load_settings()
    if cfg == CropConfig():
        d.pop("crop", None)
    else:
        d["crop"] = cfg.to_dict()
    save_settings(d)
    return SETTINGS


def clear_crop_config_cache() -> None:
    """Tests: drop the in-memory copy so the next get_ reloads from disk."""
    global _cache
    _cache = None
