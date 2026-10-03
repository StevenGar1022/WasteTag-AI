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
- [ ] Fase 2 — `src/train.py` (yolo26s, 640, batch 8, epochs 60)
- [ ] Fase 3 — `src/validate.py`
- [ ] Fase 4 — `src/auto_label.py`
- [ ] Fase 5 — `src/predict.py` demo video
- [ ] Fase 6 — docs final (`v1.0.0`)
