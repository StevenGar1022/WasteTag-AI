"""Fase 2: entrena YOLO26s con configs/base.yaml + overrides CLI.
Uso (desde raíz, dentro de env):
    ./env/bin/python src/train.py --config configs/base.yaml
    ./env/bin/python src/train.py --config configs/base.yaml --epochs 20 --batch 8
Pesos salen en runs/train/<name>/weights/best.pt (ignorado en git).
"""
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def load_cfg(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    model_ckpt = args.model or cfg.get("model", "yolo26s.pt")
    data = args.data or cfg["data"]
    epochs = args.epochs if args.epochs is not None else int(cfg.get("epochs", 20))
    imgsz = args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 640))
    batch = args.batch if args.batch is not None else int(cfg.get("batch", 8))
    patience = args.patience if args.patience is not None else int(cfg.get("patience", 20))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 2))
    project = Path(args.project or cfg.get("project", "runs/train")).resolve()
    name = args.name or cfg.get("name", "yolo26s_autos")
    device = args.device if args.device is not None else cfg.get("device", 0)

    print("=== TRAIN YOLO26s ===")
    print(f"model={model_ckpt} data={data} epochs={epochs} imgsz={imgsz} "
          f"batch={batch} device={device} patience={patience} workers={workers}")
    print(f"project={project} name={name} resume={args.resume}")

    if args.resume:
        # resume desde last.pt del run existente
        last = project / name / "weights" / "last.pt"
        print(f"resume desde {last} existe={last.exists()}")
        model = YOLO(str(last) if last.exists() else model_ckpt)
    else:
        model = YOLO(model_ckpt)
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
        resume=args.resume,
        verbose=True,
        plots=True,
        save=True,
    )
    best = project / name / "weights" / "best.pt"
    print(f"\nbest.pt: {best.resolve()} existe={best.exists()}")
    print(f"mAP50-95: {getattr(results, 'results_dict', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
