# WasteTag-AI

Auto-etiquetado + detección con **YOLO26s** (`yolo26s.pt`, Ultralytics ≥ 8.4.163).

Proyecto base actual: **autos** con 5 clases. El pipeline es reutilizable: para otro dominio (p. ej. basura) cambia las clases en `data/dataset/data.yaml` y reentrena.

> Tag actual: `v1.0.0` en `main`.

## 1. Descarga e instalación

Guía paso a paso para usuario normal. Solo necesitas una terminal con **Python ≥ 3.10** (probado en 3.14.7) y `git`. Todo queda dentro de `env/` (entorno virtual, ignorado en git).

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/StevenGar1022/WasteTag-AI.git
cd WasteTag-AI
```

### Paso 2 — Instalar todo con `./setup.sh`

```bash
./setup.sh
```

El script (`setup.sh`) hace 3 cosas, en orden:

1. `[1/3] Creando entorno virtual env/...` → ejecuta `python3 -m venv env` (si `env/` ya existe, lo reutiliza).
2. `[2/3] Instalando dependencias...` → `./env/bin/pip install -r requirements.txt` (`ultralytics>=8.4.163`, `torch`, `torchvision`, `opencv-python`, `pyyaml`, `matplotlib`, `rich`). Ojo: `torch` + CUDA pesan ~4 GB, este paso tarda varios minutos.
3. `[3/3] Registrando comando wastetag-ai...` → `./env/bin/pip install -e .` (registra el comando según `pyproject.toml` → `src.wastetag_cli:main`) y lo autoverifica con `wastetag-ai --help`.

Salida esperada al terminar:

```text
OK. Ahora ejecuta:  ./wastetag-ai
(comando wastetag-ai verificado)
```

### Paso 3 — Verificar la instalación

```bash
./wastetag-ai --help
```

Debe mostrar el uso y la descripción del programa (verificado en `src/wastetag_cli.py`):

```text
usage: wastetag-ai [-h] [--source SOURCE] [--weights WEIGHTS] ...
WASTETAG-AI — Herramienta interactiva de Auto-Etiquetado
```

Si en vez de eso ves `No existe env/. Ejecuta primero:  ./setup.sh`, es que el Paso 2 no se completó (ver mini-tabla abajo).

### Paso 4 — Ejecutar

Sesión interactiva (recomendada la primera vez):

```bash
./wastetag-ai
```

Te guía en 4 pasos: **1)** elegir modelo `.pt`, **2)** elegir carpeta o imagen, **3)** confirmar parámetros (`conf`, `imgsz`, `device`, salida), **4)** etiquetado con resumen y verificación visual. Detalle completo en la [sección 2](#2-cli-interactivo-wastetag-ai).

Ejemplo directo con las imágenes de prueba que sí vienen con el clone (`data/samples/`, 6 imágenes):

```bash
./wastetag-ai --source data/samples --weights yolo26s.pt --no-abrir
```

### Paso 5 — Poner tu propio modelo

El `git clone` **no trae ningún `.pt`**: `*.pt` y `runs/` están ignorados en `.gitignore` (no se suben al repo).

- **Tu modelo entrenado:** copia tu `best.pt` a `runs/train/yolo26s_autos/weights/best.pt` (crea las carpetas si no existen). Esa es la ruta que el CLI usa por defecto.
- **Modelo base:** `yolo26s.pt` no hace falta descargarlo a mano — Ultralytics lo descarga solo la primera vez que se usa (el CLI te lo ofrece: escribe `yolo26s.pt` cuando te pida el modelo).

### Si ves esto → haz esto

| Si ves esto | Haz esto |
|---|---|
| `No existe env/. Ejecuta primero:  ./setup.sh` (`env/` ausente) | Ejecuta `./setup.sh` completo desde la raíz del repo y reintenta |
| La instalación de `torch` tarda mucho o parece congelada | Normal: son ~4 GB; no canceles, espera con conexión estable |
| Sin GPU, error de CUDA o memoria llena  | Usa `--device cpu` y, si sigue fallando, baja a `--batch 4 --imgsz 512` |
| Sin visor gráfico o error con `xdg-open` | Añade `--no-abrir` y abre `auto_labels/contact_sheet_verificacion.jpg` manualmente |

Equivalente manual (todo dentro de `env/`):

```bash
python3 -m venv env && source env/bin/activate
./env/bin/pip install -r requirements.txt
./env/bin/pip install -e .   # registra el comando wastetag-ai
```

Verificar instalación:

```bash
./env/bin/python -c "from ultralytics import YOLO; m=YOLO('yolo26s.pt'); print('YOLO26s OK')"
wastetag-ai --help
```

Requisitos: Python ≥ 3.10 (probado en 3.14.7), `ultralytics>=8.4.163`, `torch`, `torchvision`, `opencv-python`, `pyyaml`, `matplotlib`, `rich` (ver `requirements.txt`).

## 2. CLI interactivo `wastetag-ai`

Comando registrado en `pyproject.toml` → `src.wastetag_cli:main`. Sin flags abre sesión guiada; con `--source` va en modo directo.

```bash
# Sesión interactiva paso a paso (elige .pt, carpeta, params)
./wastetag-ai

# Modo directo (catálogos grandes: 100, 60 mil...)
./wastetag-ai --source /ruta/fotos --weights runs/train/yolo26s_autos/weights/best.pt -r --lote 32
./wastetag-ai --source IMG.jpg --conf 0.5 --no-abrir
```

Flags útiles: `-r/--recursivo` (subcarpetas), `--lote 32` (inferencia por lotes, rápido),
`--sobrescribir` (rehacer `.txt` existentes; si no, se omiten), `--racha-alerta 200`,
`--muestras 4` + `--semilla 7` (verificación visual), `--no-abrir` (sin visor).

Sesión interactiva (4 pasos):

1. **Modelo (.pt):** lista los `.pt` de `runs/**/weights/*.pt`, `weights/` y `yolo26s.pt` en tabla con tamaño/fecha, o `0` para otra ruta.
2. **Imágenes:** pide carpeta o imagen; valida extensiones `.jpg/.jpeg/.png/.bmp/.webp` y muestra el total.
3. **Parámetros:** `conf` (def. 0.35), `imgsz` (def. 640), `device` (def. `0`), salida (def. `auto_labels`). Enter = aceptar.
4. **Etiquetado:** barra de progreso Rich, resumen (cobertura %, cajas/imagen, conf media, cajas por clase), diagnóstico y verificación visual.

Flags reales (`src/wastetag_cli.py`):

| Flag | Defecto | Descripción |
|---|---|---|
| `--source` | interactivo | Imagen o carpeta. Si se pasa, modo directo |
| `--weights` | `runs/train/yolo26s_autos/weights/best.pt` | Ruta al `.pt` |
| `--output` | `auto_labels` | Carpeta de labels `.txt` YOLO |
| `--conf` | `0.35` | Confianza mínima |
| `--imgsz` | `640` | Resolución de inferencia |
| `--device` | `0` | `0` GPU / `cpu` |
| `--racha-alerta` | `200` | Racha de no-etiquetadas que dispara diagnóstico |
| `--muestras` | `4` | Nº imágenes aleatorias para verificación visual |
| `--semilla` | `7` | Semilla del muestreo aleatorio |
| `--no-abrir` | — | No abrir el visor automáticamente |
| `-r/--recursivo` | off | Buscar imágenes también en subcarpetas |
| `--lote` | `32` | Tamaño de lote de inferencia (rápido para miles de imgs) |
| `--sobrescribir` | off | Re-etiquetar aunque ya exista el `.txt` (si no, se omite) |

Diagnósticos: si **0 etiquetadas** → alerta roja (conf muy alto, modelo de otro dominio, pesos/imgsz inadecuados). Si hay **racha ≥ `--racha-alerta`** → alerta amarilla (probable overfitting o desbalance de clases). Si todo OK → panel verde `AUTO ETIQUETADO EXITOSO`.

Verificación visual: re-ejecuta `--muestras` imágenes al azar y genera `contact_sheet_verificacion.jpg` en la carpeta de salida; lo abre con `xdg-open` / `open` salvo `--no-abrir`. Solo escribe `.txt` si hay detecciones ≥ conf; borra TXTs vacíos.

## 3. Scripts `src/`

Todos se ejecutan desde la raíz dentro de `env/` con `./env/bin/python src/<script>.py`.

```bash
# Fase 1 — validar dataset
./env/bin/python src/check_dataset.py --data data/dataset/data.yaml

# Fase 2 — entrenar (base.yaml + overrides)
./env/bin/python src/train.py --config configs/base.yaml
./env/bin/python src/train.py --config configs/base.yaml --epochs 20 --batch 8 --workers 0
./env/bin/python src/train.py --config configs/base.yaml --resume   # continúa desde last.pt

# Fase 3 — validar (val por defecto, métricas en runs/val/<name>/metrics.txt)
./env/bin/python src/validate.py
./env/bin/python src/validate.py --weights runs/train/yolo26s_autos/weights/best.pt --split val
./env/bin/python src/validate.py --split test

# Fase 4 — auto-etiquetar a .txt YOLO
./env/bin/python src/auto_label.py --source data/samples --weights runs/train/yolo26s_autos/weights/best.pt
./env/bin/python src/auto_label.py --source IMG.jpg --output auto_labels --conf 0.35 --imgsz 640 --device 0

# Fase 5 — demo para video (anotadas + TXTs con conf en runs/predict/demo/)
./env/bin/python src/predict.py
./env/bin/python src/predict.py --source data/samples --name demo --conf 0.35
```

Flags verificados en `argparse`:

- `check_dataset.py`: `--data` (def. `data/dataset/data.yaml`).
- `train.py`: `--config`, `--model`, `--data`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--patience`, `--project`, `--name`, `--resume`, `--workers`. Los valores CLI prevalecen sobre `configs/base.yaml` (`model: yolo26s.pt`, `epochs: 20`, `imgsz: 640`, `batch: 8`, `device: 0`, `patience: 20`, `project: runs/train`, `name: yolo26s_autos`).
- `validate.py`: `--weights`, `--data`, `--split {val,test,train}`, `--imgsz`, `--batch`, `--device`, `--conf` (def. 0.001), `--project`, `--name`.
- `auto_label.py`: `--source` (requerido), `--weights`, `--output`, `--conf`, `--imgsz`, `--device`.
- `predict.py`: `--weights`, `--source`, `--project`, `--name`, `--conf`, `--imgsz`, `--device`.

## 4. Dataset

Splits verificados en local (conteo directo):

| Split | Imágenes | Notas |
|---|---|---|
| `train` | 1516 | `data/dataset/train/images` |
| `valid` (`val`) | 10 | `data.yaml` usa `val: valid/images`; `check_dataset.py` tiene fallback `val` ↔ `valid` |
| `test` | 6 | `data/dataset/test/images` |

Clases (`nc: 5`, según `data/dataset/data.yaml`, origen Roboflow `ieee2/car-make-and-model-identification-2`, CC BY 4.0):

| ID | Clase |
|---|---|
| 0 | Mahindra XUV700 |
| 1 | Maruti Suzuki Brezza |
| 2 | Maruti Suzuki Dzire |
| 3 | Maruti Suzuki Swift |
| 4 | Toyota Fortuner |

Notas:

- `data/dataset/` está **ignorado en git** (dataset local completo). No se sube.
- `data/samples/` **sí va a git**: 6 imágenes + 6 labels copiados de `test/`, sirven de demo (`src/predict.py --source data/samples`) y de comparativa GT vs predicción para el video.
- `src/check_dataset.py` valida `nc == len(names)`, existencia de `images/` + `labels/`, conteos, labels faltantes/huérfanos, formato YOLO (`cls` en rango, coords en [0,1], `w,h > 0`) y muestra 1 ejemplo.

## 5. Resultados del modelo

Entrenamiento: YOLO26s, 640 px, batch 8, **20 epochs**, `workers 0` (ver troubleshooting), run `runs/train/yolo26s_autos/` → `weights/best.pt` (ignorado en git). Métricas de `runs/val/*/metrics.txt` (`model.val` con `conf=0.001`):

Val (10 imgs, solo clase 0 presente):

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.9952 | 1.0000 | 0.9950 | 0.9950 |

Test (6 imgs, clases 0 y 2 presentes):

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.9860 | 1.0000 | 0.9950 | 0.9950 |

Por clase (mAP50): `0 Mahindra XUV700: 0.9950` (val y test), `2 Maruti Suzuki Dzire: 0.9950` (test). Val/test son muy pequeños (10/6 imgs) → métricas casi perfectas pero **no concluyentes**; el CLI avisa de posible overfitting si aparecen rachas sin detecciones.

Auto-etiquetado y demo: 6/6 samples con detecciones, clases y cajas ≈ GT (desvío < 0.005); `src/predict.py` genera 6 cajas en `runs/predict/demo/` con labels con conf para comparar en video.

## 6. Estructura del repo

```text
wastetag-ai/
  setup.sh               # instalador (venv + deps + comando)
  wastetag-ai            # lanzador del CLI (sin activar env)
  configs/base.yaml        # hiperparámetros base (epochs 20, imgsz 640, batch 8, ...)
  data/dataset/            # IGNORADO — dataset local (1516/10/6)
  data/samples/            # SÍ a git — 6 imgs + 6 labels demo
  src/
    utils.py               # Fase 1: load yaml, splits, parse YOLO
    check_dataset.py       # Fase 1: valida data.yaml
    train.py               # Fase 2: entrena YOLO26s
    validate.py            # Fase 3: val/test + metrics.txt
    auto_label.py          # Fase 4: auto-etiquetado a .txt
    predict.py             # Fase 5: demo video (anotadas + TXTs)
    wastetag_cli.py        # Fase 4.5: CLI interactivo wastetag-ai
  runs/                    # IGNORADO — pesos, logs, val, predict
  env/                     # IGNORADO — venv
  weights/                 # .pt sueltos (ignorado por *.pt)
  yolo26s.pt               # base (ignorado por *.pt)
  requirements.txt / pyproject.toml
```

`.gitignore` (qué no se sube y por qué):

- `env/ venv/ .venv/` — entorno local, recreable con `requirements.txt`.
- `*.pt *.onnx *.engine` + `runs/` — pesos y artefactos pesados/regenerables (`best.pt`, `last.pt`, logs, `metrics.txt`, predict).
- `data/dataset/ data/raw/ data/processed/ datasets/` — dataset completo local; solo `data/samples/` va a git como demo ligera.
- `auto_labels/ **/auto_labels/` — salida generada del etiquetado.
- `__pycache__/ *.py[cod] *.egg-info/ .pytest_cache/ .mypy_cache/` — artefactos Python.
- `.DS_Store .vscode/ .idea/ *.log` — sistema/editores/logs.

## 7. Troubleshooting

- **`workers=0` en Python 3.14:** el DataLoader multiproceso falla en 3.14 → entrena con `--workers 0` (`configs/base.yaml` trae `workers: 2` por defecto; el run consolidado usó override a 0). Si ves errores de workers/dataloader, repite con `--workers 0`.
- **CUDA/GPU 4 GB:** usa `--device 0 --batch 8 --imgsz 640` (config probada). Si hay OOM, baja a `--batch 4 --imgsz 512` o `--device cpu`.
- **`xdg-open` (verificación visual):** en Linux sin entorno gráfico el mosaico no se abre solo → usa `--no-abrir` y abre `auto_labels/contact_sheet_verificacion.jpg` manualmente. Sin PIL/numpy el CLI avisa y omite el mosaico.
- **Auth GitHub:** si `git push` pide credencial, usa Personal Access Token (classic) como password o `gh auth login`; no subas el token al repo.

## 8. Changelog (tags y fases)

Historial (`git log --oneline`, `git tag`):

| Tag | Fase | Cambio |
|---|---|---|
| `v0.1.0` | Fase 0 | Estructura base, `.gitignore`, `requirements.txt`, `data/samples/` + conexión GitHub |
| `v0.2.0` | Fase 1 | `src/check_dataset.py` + `utils.py`, 5 clases autos, 1516/10/6 OK |
| `v0.3.0` | Fase 2 | `src/train.py`: YOLO26s 20 epochs batch 8 workers 0, run `yolo26s_autos` consolidado |
| `v0.4.0` | Fase 3 | `src/validate.py` + métricas val/test en `runs/val/*/metrics.txt` |
| `v0.5.0` | Fase 4 | `src/auto_label.py` portable + CLI, verificado 6/6 samples |
| `v0.5.5` | Fase 4.5 | `wastetag-ai` interactivo con Rich + diagnósticos + verificación visual (`src/wastetag_cli.py`) |
| `v0.5.6` | Fase 4.6 | `./setup.sh` + lanzador `./wastetag-ai` sin activar env; CLI robusto (reintento `.pt`, params validados, `-r`, `--lote`, omitir etiquetadas) |
| `v0.6.0` | Fase 5 | `src/predict.py` demo video + labels con conf en `runs/predict/demo/` |
| `v1.0.0` | Fase 6 | Docs final (este README) |

## 9. Licencia y notas

- Licencia **MIT** (2026, Steven Garcia) — ver `LICENSE`.
- Dataset base: Roboflow Universe `ieee2/car-make-and-model-identification-2`, licencia **CC BY 4.0** (ver `data/dataset/data.yaml`).
- Para reutilizar en otro proyecto: edita `nc`/`names` en `data/dataset/data.yaml`, valida con `check_dataset.py` y reentrena con `train.py`.
