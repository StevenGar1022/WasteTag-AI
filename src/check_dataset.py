"""Fase 1: valida data.yaml + conteos + formato YOLO.
Uso (desde raíz, dentro de env):
    ./env/bin/python src/check_dataset.py --data data/dataset/data.yaml
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_data_yaml, resolve_split_dir, images_in,
    parse_yolo_label, label_issues,
)


def check_split(yaml_path: Path, key: str, split_dir: Path, nc: int, names: list) -> bool:
    ok = True
    print(f"\n--- split '{key}': {split_dir} ---")
    if not split_dir.is_dir():
        print(f"  [ERROR] no existe carpeta {split_dir}")
        return False
    # labels/ está al lado de images/
    labels_dir = split_dir.parent / "labels"
    imgs = images_in(split_dir)
    print(f"  imágenes: {len(imgs)}")
    print(f"  labels dir: {labels_dir} (existe={labels_dir.is_dir()})")
    missing, orphans, bad_files = 0, 0, 0
    total_boxes = 0
    for img in imgs:
        txt = labels_dir / (img.stem + ".txt")
        if not txt.exists():
            missing += 1
            continue
        try:
            rows = parse_yolo_label(txt)
        except Exception as e:
            bad_files += 1
            print(f"  [ERROR] {txt.name}: {e}")
            ok = False
            continue
        if not rows:
            print(f"  [WARN] {txt.name} vacío")
        total_boxes += len(rows)
        errs = label_issues(rows, nc)
        if errs:
            bad_files += 1
            print(f"  [ERROR] {txt.name}: {errs[0]}")
            ok = False
    if labels_dir.is_dir():
        img_stems = {p.stem for p in imgs}
        orphans = sum(1 for t in labels_dir.glob("*.txt") if t.stem not in img_stems)
    print(f"  sin label: {missing} | labels huérfanos: {orphans} | archivos malformados: {bad_files}")
    print(f"  total cajas: {total_boxes}")
    # muestra 1 label
    if imgs:
        for img in imgs:
            txt = labels_dir / (img.stem + ".txt")
            if txt.exists():
                rows = parse_yolo_label(txt)
                if rows:
                    cls, x, y, w, h = rows[0]
                    cname = names[cls] if 0 <= cls < len(names) else "?"
                    print(f"  ejemplo: {txt.name} -> cls={cls} ({cname}) cx={x:.4f} cy={y:.4f} w={w:.4f} h={h:.4f}")
                break
    if missing > 0:
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset/data.yaml")
    args = ap.parse_args()
    yaml_path = Path(args.data)
    print(f"data.yaml: {yaml_path.resolve()}")
    data = load_data_yaml(yaml_path)
    nc, names = int(data["nc"]), list(data["names"])
    print(f"nc={nc}")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")
    assert nc == len(names), "nc != len(names)"
    ok_all = True
    for key in ("train", "val", "test"):
        if key not in data:
            print(f"\n--- split '{key}': ausente en yaml (skip) ---")
            continue
        d = resolve_split_dir(yaml_path, str(data[key]))
        if not d.is_dir():
            # fallback: <yaml_parent>/<key>/images y alias val<->valid
            cands = [yaml_path.parent / key / "images"]
            if key == "val":
                cands.append(yaml_path.parent / "valid" / "images")
            for c in cands:
                if c.is_dir():
                    print(f"  [WARN] '{data[key]}' no existe, usando fallback {c}")
                    d = c
                    break
        if not check_split(yaml_path, key, d, nc, names):
            ok_all = False
    print("\n===================================")
    print(" CHECK DATASET:", "OK" if ok_all else "CON ERRORES")
    print("===================================")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
