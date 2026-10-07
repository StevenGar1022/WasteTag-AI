"""Fase 4: auto-etiquetado en formato YOLO con el modelo entrenado.
Port portable del script original (rutas Windows hardcodeadas -> CLI + pathlib).
Solo escribe .txt si hay detecciones >= conf; borra TXTs vacíos.

Uso (desde raíz, dentro de env):
    ./env/bin/python src/auto_label.py --source data/samples --weights runs/train/yolo26s_autos/weights/best.pt
    ./env/bin/python src/auto_label.py --source IMG.jpg --output auto_labels --conf 0.35 --imgsz 640 --device 0
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Imagen o carpeta con imágenes")
    ap.add_argument("--weights", default="runs/train/yolo26s_autos/weights/best.pt")
    ap.add_argument("--output", default="auto_labels")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

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
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
                augment=False,
            )
        except Exception as e:
            print(f"ERROR EN INFERENCIA: {e}")
            without_det += 1
            continue

        boxes = results[0].boxes
        if len(boxes) == 0:
            print("Sin detecciones")
            without_det += 1
            continue

        txt_path = out_dir / (img_path.stem + ".txt")
        saved = 0
        with open(txt_path, "w") as f:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < args.conf:
                    continue
                cls = int(box.cls[0])
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                saved += 1
                total_boxes += 1

        if saved == 0:
            txt_path.unlink(missing_ok=True)
            without_det += 1
            print("Sin detecciones válidas")
        else:
            with_det += 1
            print(f"Detecciones guardadas: {saved}")

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
