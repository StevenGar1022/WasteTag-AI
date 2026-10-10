# WasteTag-AI

Auto-etiquetado + detección con **YOLO26s** (`yolo26s.pt`).

Proyecto base actual: **autos** — 5 clases:
`Mahindra XUV700`, `Maruti Suzuki Brezza`, `Maruti Suzuki Dzire`,
`Maruti Suzuki Swift`, `Toyota Fortuner`.
Para reutilizar en otro proyecto (ej. basura) cambia las clases en
`data/dataset/data.yaml` y reentrena.

## Estructura (Fase 0)

```text
wastetag-ai/
  configs/base.yaml
  data/dataset/   # IGNORADO en git (local, 1516 train)
  data/samples/   # SÍ a git (demo/video)
  src/            # fases 1-5
  runs/           # IGNORADO (pesos, logs)
  env/            # IGNORADO (venv)
```

## Descarga e instalación (comandos simples)

```bash
git clone https://github.com/StevenGar1022/WasteTag-AI.git
cd WasteTag-AI
python3 -m venv env && source env/bin/activate
./env/bin/pip install -r requirements.txt
./env/bin/pip install -e .   # registra el comando wastetag-ai
wastetag-ai                  # CLI interactivo de auto-etiquetado
```

## Uso (todo dentro de `env`)

```bash
# activar
source env/bin/activate

# instalar
./env/bin/pip install -r requirements.txt

# verificar
./env/bin/python -c "from ultralytics import YOLO; m=YOLO('yolo26s.pt'); print('YOLO26s OK')"
```

## Fases

- [x] Fase 0 — base + samples + conexión GitHub (`v0.1.0`)
- [x] Fase 1 — `src/check_dataset.py` (valida data.yaml, 1516/10/6 OK)
- [x] Fase 2 — `src/train.py` (yolo26s, 640, batch 8, epochs 20, workers 0)
  - Pesos: `runs/train/yolo26s_autos/weights/best.pt` (ignorado en git)
  - Val (10 imgs): mAP50=0.995, mAP50-95=0.995
- [x] Fase 3 — `src/validate.py` (val + test, métricas en `runs/val/*/metrics.txt`)
  - val (10 imgs, solo clase 0): P=0.9952 R=1.0 mAP50=0.995 mAP50-95=0.995
  - test (6 imgs, clases 0 y 2): P=0.9860 R=1.0 mAP50=0.995 mAP50-95=0.995
- [x] Fase 4 — `src/auto_label.py` (port del script original a CLI portable)
  - 6/6 samples con detecciones, clases y cajas ≈ GT (desvío <0.005)
- [x] Fase 5 — `src/predict.py` demo video (anotadas + TXTs en `runs/predict/demo/`)
  - 6/6 samples, 6 cajas, clases ≈ GT (labels con conf para comparar en video)
- [x] Fase 4.5 — `wastetag-ai` CLI interactivo (`src/wastetag_cli.py`, `v0.5.5`)
  - Banner, selector de `.pt`, parámetros, barra de progreso, resumen,
    alertas (0 etiquetadas / racha 200: posible overfitting o desbalance),
    verificación visual aleatoria con contact-sheet
- [ ] Fase 6 — docs final (`v1.0.0`)
