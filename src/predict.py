"""Fase 5: demo para el video de GitHub.
Inferencia sobre data/samples -> imágenes anotadas + TXTs en runs/predict/<name>/.
Sirve para grabar la comparativa GT (data/samples/*.txt) vs predicción.

Uso (desde raíz, dentro de env):
    ./env/bin/python src/predict.py
    ./env/bin/python src/predict.py --source data/samples --name demo --conf 0.35
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from utils import resolver_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/yolo26s_autos/weights/best.pt")
    ap.add_argument("--source", default="data/samples")
    ap.add_argument("--project", default="runs/predict")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.device = resolver_device(args.device)

    project = Path(args.project).resolve()
    print("=== PREDICT DEMO ===")
    print(f"weights: {args.weights}")
    print(f"source:  {Path(args.source).resolve()}")
    print(f"salida:  {project / args.name}")
    print(f"conf={args.conf} imgsz={args.imgsz} device={args.device}\n")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        verbose=False,
    )

    n_box = sum(len(r.boxes) for r in results)
    print(f"\nimágenes: {len(results)} | cajas: {n_box}")
    print(f"anotadas + labels en: {(project / args.name).resolve()}")
    print("Compara con GT en data/samples/*.txt para el video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
