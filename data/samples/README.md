# Samples — demo / video GitHub

6 imágenes + 6 labels (formato YOLO) copiados de `data/dataset/test/`.

Clases (índice -> nombre, según `data/dataset/data.yaml`):
- 0: Mahindra XUV700
- 1: Maruti Suzuki Brezza
- 2: Maruti Suzuki Dzire
- 3: Maruti Suzuki Swift
- 4: Toyota Fortuner

Uso en Fase 5 (demo):
`env/bin/python src/predict.py --weights runs/train/yolo26s_autos/weights/best.pt --source data/samples`
