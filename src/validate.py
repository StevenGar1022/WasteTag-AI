"""Fase 3: valida el modelo entrenado y guarda métricas.
Uso (desde raíz, dentro de env):
    ./env/bin/python src/validate.py
    ./env/bin/python src/validate.py --weights runs/train/yolo26s_autos/weights/best.pt --split val
    ./env/bin/python src/validate.py --split test
Métricas en runs/val/<name>/metrics.txt (ignorado en git).
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/yolo26s_autos/weights/best.pt")
    ap.add_argument("--data", default="data/dataset/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--project", default="runs/val")
    ap.add_argument("--name", default="yolo26s_autos")
    args = ap.parse_args()

    project = Path(args.project).resolve()
    print("=== VALIDATE ===")
    print(f"weights={args.weights} data={args.data} split={args.split} "
          f"imgsz={args.imgsz} batch={args.batch} device={args.device}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        project=str(project),
        name=args.name,
        exist_ok=True,
        plots=True,
        save_json=False,
        verbose=True,
    )

    d = metrics.results_dict
    out_dir = project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"weights: {args.weights}",
        f"data: {args.data}  split: {args.split}",
        "",
        "== métricas ==",
    ]
    for k in ("metrics/precision(B)", "metrics/recall(B)",
              "metrics/mAP50(B)", "metrics/mAP50-95(B)",
              "fitness"):
        if k in d:
            lines.append(f"{k}: {d[k]:.4f}")
    lines += ["", "== por clase (mAP50) =="]
    try:
        names = metrics.names
        ap50 = metrics.box.ap50
        idx = list(getattr(metrics.box, "ap_class_index", range(len(ap50))))
        for ci, v in zip(idx, ap50):
            lines.append(f"{ci} {names.get(int(ci), '?')}: {v:.4f}")
    except Exception as e:
        lines.append(f"(detalle por clase no disponible: {e})")
    txt = "\n".join(lines) + "\n"
    print("\n" + txt)
    (out_dir / "metrics.txt").write_text(txt)
    print(f"metrics.txt: {(out_dir / 'metrics.txt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
