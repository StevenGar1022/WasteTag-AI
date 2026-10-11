"""Utilidades compartidas WasteTag-AI (Fase 1)."""
from pathlib import Path
import yaml

VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_data_yaml(yaml_path: str | Path) -> dict:
    p = Path(yaml_path)
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return data


def resolve_split_dir(yaml_path: str | Path, split_path: str) -> Path:
    """Resuelve train/val/test relativos al data.yaml (ej. ../train/images)."""
    base = Path(yaml_path).parent
    cand = Path(split_path)
    if not cand.is_absolute():
        cand = (base / cand).resolve()
    return cand


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS
    )


def parse_yolo_label(txt_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            rows.append((cls, x, y, w, h))
    return rows


def label_issues(rows: list[tuple], nc: int) -> list[str]:
    errs = []
    for cls, x, y, w, h in rows:
        if not (0 <= cls < nc):
            errs.append(f"cls {cls} fuera de [0,{nc-1}]")
        for v in (x, y, w, h):
            if not (0.0 <= v <= 1.0):
                errs.append(f"coord {v} fuera de [0,1]")
                break
        if w <= 0 or h <= 0:
            errs.append(f"w,h inválidos ({w},{h})")
    return errs


def resolver_device(valor="auto"):
    """'auto' -> '0' (CUDA), 'mps' (Apple Silicon) o 'cpu'. Pasa el resto tal cual."""
    if valor is None or str(valor).lower() == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
            if getattr(torch.backends, "mps", None) is not None \
                    and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return valor
