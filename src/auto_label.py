"""Fase 4: auto-etiquetado en formato YOLO con el modelo entrenado.
Port portable del script original (rutas Windows hardcodeadas -> CLI + pathlib).
Solo escribe .txt si hay detecciones >= conf; borra TXTs vacíos.

Uso (desde raíz, dentro de env):
    ./env/bin/python src/auto_label.py --source data/samples --weights runs/train/yolo26s_autos/weights/best.pt
    ./env/bin/python src/auto_label.py --source IMG.jpg --output auto_labels --conf 0.35 --imgsz 640 --device 0
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import resolver_device  # noqa: E402

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(source: Path, recursivo: bool = False) -> list[Path]:
    if source.is_file():
        return [source]
    if recursivo:
        return sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def escribir_txt(out_dir: Path, stem: str, dets: list) -> Path:
    txt_path = out_dir / (stem + ".txt")
    with open(txt_path, "w") as f:
        for d in dets:
            x, y, w, h = d["xywhn"]
            f.write(f"{d['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return txt_path


def etiquetar_lote(model, img_paths: list, out_dir: Path,
                   conf_min: float, imgsz: int = 640, device="0") -> list[dict]:
    """Infiere un lote de una sola pasada (rápido para miles de imágenes).
    Retorna un dict por imagen: {saved, boxes, n_raw, error}.
    """
    results = model.predict(
        source=[str(p) for p in img_paths],
        conf=conf_min,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
        augment=False,
    )
    out = []
    for img_path, res in zip(img_paths, results):
        dets = []
        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < conf_min:
                continue
            dets.append({
                "cls": int(box.cls[0]),
                "conf": conf,
                "xywhn": box.xywhn[0].tolist(),
            })
        txt_path = out_dir / (img_path.stem + ".txt")
        if dets:
            escribir_txt(out_dir, img_path.stem, dets)
            out.append({"saved": len(dets), "boxes": dets,
                        "n_raw": len(res.boxes), "error": None})
        else:
            txt_path.unlink(missing_ok=True)
            out.append({"saved": 0, "boxes": [],
                        "n_raw": len(res.boxes), "error": None})
    return out


def etiquetar_imagen(model, img_path: Path, out_dir: Path,
                     conf_min: float, imgsz: int = 640, device="0") -> dict:
    """Infiere una imagen y escribe su .txt YOLO. Reutilizable por el CLI.
    Retorna dict(saved, boxes=[{cls, conf, xywhn}], error=None|str).
    """
    res = model.predict(
        source=str(img_path),
        conf=conf_min,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
        augment=False,
    )[0]
    dets = []
    n_raw = len(res.boxes)
    for box in res.boxes:
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        dets.append({
            "cls": int(box.cls[0]),
            "conf": conf,
            "xywhn": box.xywhn[0].tolist(),
        })
    txt_path = out_dir / (img_path.stem + ".txt")
    if not dets:
        txt_path.unlink(missing_ok=True)
        return {"saved": 0, "boxes": [], "n_raw": n_raw, "error": None}
    with open(txt_path, "w") as f:
        for d in dets:
            x, y, w, h = d["xywhn"]
            f.write(f"{d['cls']} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return {"saved": len(dets), "boxes": dets, "n_raw": n_raw, "error": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Imagen o carpeta con imágenes")
    ap.add_argument("--weights", default="runs/train/yolo26s_autos/weights/best.pt")
    ap.add_argument("--output", default="auto_labels")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.device = resolver_device(args.device)

    source, out_dir = Path(args.source), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===================================")
    print(" AUTO-ETIQUETADO YOLO")
    print("===================================")
    print(f"weights: {args.weights}")
    print(f"source:  {source.resolve()}")
    print(f"output:  {out_dir.resolve()}")
    print(f"conf={args.conf} imgsz={args.imgsz} device={args.device}\n")

    model = YOLO(args.weights)
    try:
        names = model.names
        print(f"clases: {[names[i] for i in sorted(names)]}")
    except Exception:
        pass

    images = collect_images(source)
    total = len(images)
    print(f"\nTotal imágenes: {total}")
    if total == 0:
        return 1

    with_det, without_det, total_boxes = 0, 0, 0
    for i, img_path in enumerate(images):
        print(f"\n[{i+1}/{total}] {img_path.name}")
        try:
            r = etiquetar_imagen(model, img_path, out_dir,
                                 args.conf, args.imgsz, args.device)
        except Exception as e:
            print(f"ERROR EN INFERENCIA: {e}")
            without_det += 1
            continue

        if r["saved"] == 0:
            without_det += 1
            print("Sin detecciones" if r["n_raw"] == 0 else "Sin detecciones válidas")
        else:
            with_det += 1
            total_boxes += r["saved"]
            print(f"Detecciones guardadas: {r['saved']}")

    print("\n===================================")
    print(" AUTO ETIQUETADO FINALIZADO ")
    print("===================================")
    print(f"\nImágenes procesadas: {total}")
    print(f"Imágenes con detecciones: {with_det}")
    print(f"Imágenes sin detecciones: {without_det}")
    print(f"Total cajas generadas: {total_boxes}")
    print(f"\nLabels guardados en:\n{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
