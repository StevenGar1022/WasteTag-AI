#!/usr/bin/env bash
# WASTETAG-AI — instalador en un solo comando.
# Uso: ./setup.sh
# Crea env/, instala dependencias y registra el comando wastetag-ai.
set -e
cd "$(dirname "$0")"

echo "=== WASTETAG-AI setup ==="
if [ ! -x env/bin/python ]; then
    echo "[1/3] Creando entorno virtual env/..."
    python3 -m venv env
else
    echo "[1/3] env/ ya existe, se reutiliza."
fi
echo "[2/3] Instalando dependencias..."
./env/bin/pip install -r requirements.txt
echo "[3/3] Registrando comando wastetag-ai..."
./env/bin/pip install -e .
echo ""
echo "OK. Ahora ejecuta:  ./wastetag-ai"
./env/bin/wastetag-ai --help > /dev/null && echo "(comando wastetag-ai verificado)"
